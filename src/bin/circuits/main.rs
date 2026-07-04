//! Library-learning runner for egg-stitch's EPFL boolean-circuit benchmark.
//!
//! Reads a corpus of circuit s-expressions (one per line, the `.bab` format),
//! optionally applies the factoring domain-specific rewrites, runs babble's
//! beam-search lib learning, and dumps the original/rewritten programs plus the
//! learned abstractions as JSON. Modelled on the `molecules` binary, with the
//! boolean-gate grammar in `lang` swapped in for the atom-tree one.
//!
//! The DSR file must use plain `=>` rules; expand any bidirectional `<=>` rules
//! into separate forward/backward rules beforehand (egg-stitch's
//! `and_or_demorgan_factor.rewrites` uses `<=>`).

use babble::{
    ast_node::{combine_exprs, AstNode, Expr, Pretty},
    experiments::{plumbing, BeamExperiment, Experiment, Rounds},
    extract::beam::PartialLibCost,
    rewrites, util,
    sexp::Program,
};
use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use serde_json::json;
use std::{
    convert::TryInto,
    fs,
    io::{self, Read},
    path::PathBuf,
    time::Instant,
};

mod lang;
use lang::Circuit;

#[derive(Parser)]
#[clap(version, author, about)]
struct Opts {
    /// The file with training programs. If no file is specified, reads from stdin.
    #[clap(parse(from_os_str))]
    file: Option<PathBuf>,

    /// CSV path to output stuff to.
    #[clap(long, default_value = "harness/data_gen/res_circuits.csv")]
    output: String,

    /// Whether to learn "library functions" with no arguments.
    #[clap(long)]
    learn_constants: bool,

    /// Optional file with domain-specific rewrites (plain `=>` rules only).
    #[clap(long)]
    dsr: Option<PathBuf>,

    /// Maximum arity of functions to learn.
    #[clap(long)]
    max_arity: Option<usize>,

    /// The beam size to use for the beam extractor.
    #[clap(long, default_value_t = 400)]
    beams: usize,

    /// The number of libs to learn at a time.
    #[clap(long, default_value_t = 1)]
    lps: usize,

    /// The number of rounds of lib learning to run.
    #[clap(long, default_value_t = 1)]
    rounds: usize,

    /// The number of programs to use.
    #[clap(long)]
    limit: Option<usize>,

    /// Dump original/rewritten programs and learned abstractions as JSON here.
    #[clap(long)]
    dump_json: Option<PathBuf>,
}

fn main() {
    env_logger::init();
    let opts: Opts = Opts::parse();

    let input = opts
        .file
        .clone()
        .map_or_else(
            || {
                let mut buf = String::new();
                io::stdin().read_to_string(&mut buf).map(|_| buf)
            },
            fs::read_to_string,
        )
        .expect("Error reading input");

    let mut prog: Vec<Expr<Circuit>> = Program::parse(&input)
        .expect("Failed to parse training set")
        .0
        .into_iter()
        .map(|x| {
            x.try_into()
                .expect("Training input is not a valid list of expressions")
        })
        .collect();

    if let Some(limit) = opts.limit {
        prog.truncate(limit);
    }

    let initial_expr: RecExpr<_> = combine_exprs(prog.clone());
    let initial_pretty_cost = AstSize.cost_rec(&initial_expr);
    println!("Training expression (cost {initial_pretty_cost}):");
    println!("{}", Pretty(&Expr::from(initial_expr)));
    println!();

    let dsrs = if let Some(dsr_path) = opts.dsr.clone() {
        match rewrites::from_file(dsr_path) {
            Ok(dsrs) => dsrs,
            Err(e) => {
                eprintln!("Error reading dsr file: {e}");
                std::process::exit(1);
            }
        }
    } else {
        vec![]
    };
    println!("loaded {} domain-specific rewrites", dsrs.len());

    run_with_json_dump(prog, dsrs, &opts);
}

/// Runs a single beam-search experiment configuration end-to-end and, if
/// `--dump-json` is set, writes the original/rewritten programs and learned
/// abstractions as JSON (mirrors the `molecules` binary's dump format).
fn run_with_json_dump(
    prog: Vec<Expr<Circuit>>,
    dsrs: Vec<Rewrite<AstNode<Circuit>, PartialLibCost>>,
    opts: &Opts,
) {
    let beam = opts.beams;
    let beam_exp = BeamExperiment::new(
        dsrs,
        beam,
        beam,
        opts.lps,
        (),
        opts.learn_constants,
        opts.max_arity,
        1,
    );
    let rounds_exp = Rounds::new(opts.rounds, beam_exp);

    let csv_file = fs::File::create(&opts.output).expect("Failed to create CSV output");
    let boxed: Box<dyn io::Write> = Box::new(csv_file);
    let mut writer: babble::experiments::CsvWriter = csv::Writer::from_writer(boxed);

    // `Experiments::run_csv` adds 1 for the implicit list root; mirror that.
    let initial_cost = prog.iter().map(Expr::len).sum::<usize>() + 1;
    let start = Instant::now();
    let result = rounds_exp.run(prog.clone(), &mut writer);
    let elapsed = start.elapsed();
    let final_cost = result.final_expr.len();
    let compression = util::compression_factor(initial_cost, final_cost);

    rounds_exp.write_to_csv(
        &mut writer,
        rounds_exp.total_rounds(),
        initial_cost,
        final_cost,
        compression,
        result.num_libs,
        elapsed,
    );

    println!(
        "initial cost {initial_cost}, final cost {final_cost}, \
         compression {compression:.4}, {} libs, {:.2}s",
        result.num_libs,
        elapsed.as_secs_f64(),
    );

    let Some(json_path) = opts.dump_json.clone() else {
        return;
    };

    let final_recexpr: RecExpr<AstNode<Circuit>> = result.final_expr.clone().into();
    let libs_map = plumbing::libs(final_recexpr.as_ref());
    let rewritten_exprs = plumbing::exprs(final_recexpr.as_ref());

    let original: Vec<String> = prog
        .iter()
        .map(|e| RecExpr::<AstNode<Circuit>>::from(e.clone()).to_string())
        .collect();
    let rewritten: Vec<String> = rewritten_exprs
        .into_iter()
        .map(|e| RecExpr::<AstNode<Circuit>>::from(e).to_string())
        .collect();

    let mut sorted_libs: Vec<_> = libs_map.into_iter().collect();
    sorted_libs.sort_by_key(|(id, _)| id.0);
    let abstractions: Vec<_> = sorted_libs
        .into_iter()
        .map(|(id, body_nodes)| {
            let r: RecExpr<AstNode<Circuit>> = body_nodes.into();
            json!({ "id": id.0, "body": r.to_string() })
        })
        .collect();

    let dump = json!({
        "elapsed_secs": elapsed.as_secs_f64(),
        "rounds": opts.rounds,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "compression": compression,
        "original": original,
        "rewritten": rewritten,
        "abstractions": abstractions,
    });
    fs::write(&json_path, serde_json::to_string_pretty(&dump).unwrap())
        .expect("Failed to write JSON dump");
    println!("wrote JSON dump to {}", json_path.display());
}
