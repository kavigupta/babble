#![warn(
    clippy::all,
    clippy::pedantic,
    anonymous_parameters,
    elided_lifetimes_in_paths,
    missing_copy_implementations,
    missing_debug_implementations,
    single_use_lifetimes,
    trivial_casts,
    unreachable_pub,
    unused_lifetimes
)]
#![allow(clippy::non_ascii_literal)]

use babble::{
    ast_node::{AstNode, Expr},
    dreamcoder::{expr::DreamCoderOp, json::CompressionInput},
    experiments::{cache::Cache, plumbing, BeamExperiment, EqsatExperiment, Experiment, Rounds, Summary},
    extract::beam::PartialLibCost,
    rewrites, util,
};
use clap::Parser;
use egg::{RecExpr, Rewrite};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use rayon::prelude::*;

#[allow(clippy::struct_excessive_bools)]
#[derive(Parser)]
#[clap(version, author, about)]
struct Opts {
    /// The input directory. If none is specified, defaults to `"harness/data/dreamcoder-benchmarks/benches"`.
    #[clap(parse(from_os_str))]
    file: Option<PathBuf>,

    #[clap(long)]
    domain: Option<String>,

    #[clap(long)]
    cache: Option<PathBuf>,

    /// File to dump the raw costs into
    #[clap(long, short)]
    output: PathBuf,

    #[clap(long)]
    beam_size: usize,
    #[clap(long)]
    lps: usize,
    #[clap(long)]
    rounds: usize,
    #[clap(long)]
    max_arity: usize,
    #[clap(long)]
    lib_iter_limit: usize,
    #[clap(long)] // should be bool, but I don't want flags
    use_all: usize,
    #[clap(long, value_parser = ["babble", "au", "eqsat"])]
    mode: String,

    /// Dump parsed programs as JSON and exit (for egg-stitch compatibility)
    #[clap(long)]
    dump: Option<PathBuf>,

    /// After running, dump original/rewritten programs and learned abstractions
    /// as JSON to this path. Requires `--domain` so a single domain's results
    /// land in one file.
    #[clap(long)]
    dump_json: Option<PathBuf>,

    /// Run on a single CompressionInput JSON file instead of walking the
    /// benchmark directory tree. Requires `--domain` (used to locate the
    /// matching `<domain>.rewrites` file). The file's basename is used as
    /// the `file` metadata field; its parent directory name as `benchmark`.
    #[clap(long)]
    input_file: Option<PathBuf>,
}

const BENCHMARK_PATH: &str = "harness/data/dreamcoder-benchmarks/benches";
const DSR_PATH: &str = "harness/data/benchmark-dsrs";

/// Format an Expr<DreamCoderOp> for egg-stitch: lambda -> lam, Inlined -> named atoms.
fn format_for_stitch(
    expr: &Expr<DreamCoderOp>,
    inlined_names: &mut BTreeMap<String, String>,
) -> String {
    let node = expr.0.operation();
    let args = expr.0.args();
    match (node, args) {
        (DreamCoderOp::Symbol(s), []) => format!("{s}"),
        (DreamCoderOp::Var(i), []) => format!("${i}"),
        (DreamCoderOp::Inlined(inner), []) => {
            let key = format!("{}", babble::dreamcoder::expr::DcExpr::from((**inner).clone()));
            let n = inlined_names.len();
            inlined_names
                .entry(key)
                .or_insert_with(|| format!("fn_{n}"))
                .clone()
        }
        (DreamCoderOp::Lambda, [body]) => {
            format!("(lam {})", format_for_stitch(body, inlined_names))
        }
        (DreamCoderOp::App, [fun, arg]) => {
            // Flatten nested applications: (((f a) b) c) -> (f a b c)
            let mut head = fun;
            let mut args_rev = vec![arg];
            while let (DreamCoderOp::App, [inner_fun, inner_arg]) =
                (head.0.operation(), head.0.args())
            {
                args_rev.push(inner_arg);
                head = inner_fun;
            }
            let mut s = format!("({}", format_for_stitch(head, inlined_names));
            for a in args_rev.into_iter().rev() {
                s.push(' ');
                s.push_str(&format_for_stitch(a, inlined_names));
            }
            s.push(')');
            s
        }
        (op, children) => {
            let mut s = format!("({op}");
            for c in children {
                s.push(' ');
                s.push_str(&format_for_stitch(c, inlined_names));
            }
            s.push(')');
            s
        }
    }
}

#[derive(Debug)]
struct Benchmark<'a> {
    name: &'a str,
    path: &'a Path,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Iteration {
    domain: String,
    benchmark: String,
    file: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Compression {
    initial_size: usize,
    final_size: usize,
    run_time: f32,
}

impl<'a, Op> From<&'a Summary<Op>> for Compression {
    fn from(summary: &'a Summary<Op>) -> Self {
        Self {
            initial_size: summary.initial_cost,
            final_size: summary.final_cost,
            run_time: summary.run_time.as_secs_f32(),
        }
    }
}

impl<'a, Op> From<&'a Option<Summary<Op>>> for Compression {
    fn from(summary: &'a Option<Summary<Op>>) -> Self {
        Self {
            initial_size: summary.as_ref().map_or_else(|| 1, |x| x.initial_cost),
            final_size: summary.as_ref().map_or_else(|| 1, |x| x.final_cost),
            run_time: summary
                .as_ref()
                .map_or_else(|| 0.0, |x| x.run_time.as_secs_f32()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchResults {
    domain: String,
    benchmark: String,
    file: String,
    summary: Summary<DreamCoderOp>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let opts: Opts = Opts::parse();

    let cache = opts
        .cache
        .clone()
        .map_or_else(Cache::new, Cache::from_dir)?;

    println!("using cache: {}", cache.path().to_str().unwrap());

    let cache = Mutex::new(cache);

    let benchmark_path = opts.file.clone().unwrap_or(PathBuf::from(BENCHMARK_PATH));

    let mut benchmark_dirs = Vec::new();
    for entry in fs::read_dir(&benchmark_path)? {
        let path = entry?.path();
        if fs::metadata(&path)?.is_dir() {
            benchmark_dirs.push(path);
        }
    }

    benchmark_dirs.sort_unstable();

    let mut domains: BTreeMap<_, Vec<_>> = BTreeMap::new();

    for benchmark_dir in &benchmark_dirs {
        let dir_name = benchmark_dir.file_name().unwrap().to_str().unwrap();
        let (domain, benchmark_name) = dir_name.split_once('_').unwrap();
        domains.entry(domain).or_default().push(Benchmark {
            name: benchmark_name,
            path: benchmark_dir.as_path(),
        });
    }

    println!("domains:");
    for (domain, benchmarks) in &domains {
        println!("  {domain}: {} benchmark(s)", benchmarks.len());
    }

    if let Some(dump_dir) = &opts.dump {
        let target_domains: Vec<_> = if let Some(domain) = &opts.domain {
            vec![(domain.as_str(), domains[domain.as_str()].as_slice())]
        } else {
            domains.iter().map(|(d, bs)| (*d, bs.as_slice())).collect()
        };

        let mut inlined_names = BTreeMap::new();
        for (domain, benchmarks) in target_domains {
            let domain_dir = dump_dir.join(domain);
            fs::create_dir_all(&domain_dir).unwrap();

            for benchmark in benchmarks {
                let mut inputs = Vec::new();
                for entry in fs::read_dir(benchmark.path).unwrap() {
                    let path = entry.unwrap().path();
                    if fs::metadata(&path).unwrap().is_file() {
                        inputs.push(path);
                    }
                }
                inputs.sort();

                for input_path in &inputs {
                    let file = input_path.file_name().unwrap().to_str().unwrap();
                    let raw = fs::read_to_string(input_path).unwrap();
                    let input: CompressionInput = serde_json::from_str(&raw).unwrap();

                    let mut programs = Vec::new();
                    for frontier in &input.frontiers {
                        for p in &frontier.programs {
                            let expr: Expr<DreamCoderOp> = p.program.clone().into();
                            programs.push(format_for_stitch(&expr, &mut inlined_names));
                        }
                    }

                    let out_name = format!("{}__{}", benchmark.name, file);
                    let out_path = domain_dir.join(&out_name);
                    let json = serde_json::to_string_pretty(&programs).unwrap();
                    fs::write(&out_path, json).unwrap();
                    println!("{}: {} programs", out_path.display(), programs.len());
                }
            }
        }

        return Ok(());
    }

    if let Some(input_file) = opts.input_file.clone() {
        let domain = opts
            .domain
            .as_ref()
            .expect("--input-file requires --domain to locate the matching DSR file");
        run_single_file(domain, &opts, &input_file);
        return Ok(());
    }

    if let Some(domain) = &opts.domain {
        run_domain(domain, &opts, &domains[domain.as_str()], &cache);
    } else {
        assert!(
            opts.dump_json.is_none(),
            "--dump-json requires --domain so a single run lands in one file"
        );
        for (domain, benchmarks) in domains {
            run_domain(domain, &opts, &benchmarks, &cache);
        }
    }

    Ok(())
}

fn run_domain(
    domain: &str,
    opts: &Opts,
    benchmarks: &[Benchmark<'_>],
    _cache: &Mutex<Cache<DreamCoderOp>>,
) {
    let results = Mutex::new(Vec::new());

    println!("domain: {domain}");

    let dsr_file = PathBuf::from(DSR_PATH).join(format!("{domain}.rewrites"));
    let rewrites = rewrites::try_from_file(dsr_file)
        .unwrap()
        .unwrap_or_default();

    println!("  found {} domain-specific rewrites", rewrites.len());

    benchmarks.par_iter().for_each(|benchmark| {
        println!("  benchmark: {}", benchmark.name);
        let mut inputs = Vec::new();

        for entry in fs::read_dir(benchmark.path).unwrap() {
            let path = entry.unwrap().path();
            if fs::metadata(&path).unwrap().is_file() {
                inputs.push(path);
            }
        }

        inputs.sort();

        inputs.par_iter().for_each(|input| {
            let bench_results = run_one_file(domain, benchmark.name, input, &rewrites, opts);
            let mut locked = results.lock().unwrap();
            locked.push(bench_results);
        });
    });

    let results = results.into_inner().unwrap();
    plot_raw_data(&results, opts).unwrap();

    if let Some(json_path) = &opts.dump_json {
        write_json_dump(&results, json_path);
    }
}

/// Runs the experiment on a single `CompressionInput` JSON file. Shared
/// between the directory-walking `run_domain` and the `--input-file` mode
/// so both paths use the same per-file pipeline.
fn run_one_file(
    domain: &str,
    benchmark_name: &str,
    input_path: &Path,
    rewrites: &[Rewrite<AstNode<DreamCoderOp>, PartialLibCost>],
    opts: &Opts,
) -> BenchResults {
    let file = input_path.file_name().unwrap().to_str().unwrap();
    println!("    file: {file}");

    let raw = fs::read_to_string(input_path).unwrap();
    let input: CompressionInput = serde_json::from_str(&raw).unwrap();

    let program_groups: Vec<Vec<Expr<_>>> = input
        .frontiers
        .iter()
        .cloned()
        .map(|frontier| -> Vec<Expr<DreamCoderOp>> {
            let programs = frontier
                .programs
                .into_iter()
                .map(|program| program.program.into());

            if opts.use_all > 0 {
                programs.collect()
            } else {
                programs.take(1).collect()
            }
        })
        .collect();

    let summary = if opts.mode == "eqsat" {
        let experiment = Rounds::new(1, EqsatExperiment::new(rewrites.to_vec(), ()));
        experiment.run_multi_summary(program_groups)
    } else {
        let use_dsrs = match opts.mode.as_str() {
            "babble" => true,
            "au" => false,
            m => panic!("bad mode: {}", m),
        };
        let experiment = Rounds::new(
            opts.rounds,
            BeamExperiment::new(
                if use_dsrs { rewrites.to_vec() } else { vec![] },
                opts.beam_size,
                opts.beam_size,
                opts.lps,
                (),
                true,
                Some(opts.max_arity),
                opts.lib_iter_limit,
            ),
        );
        experiment.run_multi_summary(program_groups)
    };

    let name = format!("{domain}_{benchmark_name}/{file}");
    println!(
        "{name:20}        {} -> {} (r {:.3}), with {:>3} libs in {:>8.3}s",
        summary.initial_cost,
        summary.final_cost,
        util::compression_factor(summary.initial_cost, summary.final_cost),
        summary.num_libs,
        summary.run_time.as_secs_f32(),
    );

    BenchResults {
        domain: domain.to_string(),
        benchmark: benchmark_name.to_string(),
        file: file.to_string(),
        summary,
    }
}

/// Runs the experiment on a single CompressionInput file (no benchmark-tree
/// walking). The DSRs are still loaded by domain name from the standard
/// `<DSR_PATH>/<domain>.rewrites` location. Mirrors `run_domain` but for
/// exactly one input.
fn run_single_file(domain: &str, opts: &Opts, input_file: &Path) {
    println!("domain: {domain}");

    let dsr_file = PathBuf::from(DSR_PATH).join(format!("{domain}.rewrites"));
    let rewrites = rewrites::try_from_file(dsr_file)
        .unwrap()
        .unwrap_or_default();

    println!("  found {} domain-specific rewrites", rewrites.len());

    // Match the directory-walk convention: parent dir is `<domain>_<bench>`,
    // and the existing code strips the `<domain>_` prefix. Fall back to the
    // raw dir name (or "single") if the convention doesn't apply.
    let parent_name = input_file
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("single");
    let benchmark_name = parent_name
        .strip_prefix(&format!("{domain}_"))
        .unwrap_or(parent_name);

    let bench_results = run_one_file(domain, benchmark_name, input_file, &rewrites, opts);
    let results = vec![bench_results];

    plot_raw_data(&results, opts).unwrap();

    if let Some(json_path) = &opts.dump_json {
        write_json_dump(&results, json_path);
    }
}

/// Dumps per-file original/rewritten programs and learned abstractions as JSON.
/// Programs are formatted via `format_for_stitch` so identifiers match what the
/// `--dump` mode emits, keeping downstream consumers (e.g. `egg-stitch`'s
/// `run_babble`) on a single naming convention.
fn write_json_dump(results: &[BenchResults], json_path: &Path) {
    let mut inlined_names: BTreeMap<String, String> = BTreeMap::new();
    let mut files = Vec::new();
    for r in results {
        let summary = &r.summary;
        let final_recexpr: RecExpr<AstNode<DreamCoderOp>> = summary.final_expr.clone().into();
        let libs_map = plumbing::libs(final_recexpr.as_ref());
        let rewritten_exprs = plumbing::exprs(final_recexpr.as_ref());

        // Flatten initial_expr_groups: with --use-all=0 each group is a single
        // program; with --use-all>0 every program in the frontier is included.
        let original: Vec<String> = summary
            .initial_expr_groups
            .iter()
            .flat_map(|group| group.iter())
            .map(|e| format_for_stitch(e, &mut inlined_names))
            .collect();

        let rewritten: Vec<String> = rewritten_exprs
            .iter()
            .map(|e| format_for_stitch(e, &mut inlined_names))
            .collect();

        let mut sorted_libs: Vec<_> = libs_map.into_iter().collect();
        sorted_libs.sort_by_key(|(id, _)| id.0);
        let abstractions: Vec<_> = sorted_libs
            .into_iter()
            .map(|(id, body_nodes)| {
                let r: RecExpr<AstNode<DreamCoderOp>> = body_nodes.into();
                let body_expr: Expr<DreamCoderOp> = r.into();
                json!({
                    "id": id.0,
                    "body": format_for_stitch(&body_expr, &mut inlined_names),
                })
            })
            .collect();

        files.push(json!({
            "domain": r.domain,
            "benchmark": r.benchmark,
            "file": r.file,
            "elapsed_secs": summary.run_time.as_secs_f64(),
            "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost,
            "num_libs": summary.num_libs,
            "original": original,
            "rewritten": rewritten,
            "abstractions": abstractions,
        }));
    }
    let dump = json!({ "files": files });
    fs::write(json_path, serde_json::to_string_pretty(&dump).unwrap())
        .expect("Failed to write JSON dump");
    println!("wrote JSON dump to {}", json_path.display());
}

#[allow(clippy::cast_precision_loss)]
fn plot_raw_data(results: &[BenchResults], opts: &Opts) -> anyhow::Result<()> {
    let mut csv_writer = csv::Writer::from_path(&opts.output)?;
    csv_writer.serialize((
        "name",
        "iter",
        "initial cost",
        "final cost",
        "compression",
        "total time",
        "num libs",
    ))?;

    for BenchResults {
        domain,
        benchmark,
        file,
        summary:
            Summary {
                initial_cost,
                final_cost,
                num_libs,
                run_time,
                ..
            },
    } in results
    {
        csv_writer.serialize((
            format!("{domain}_{benchmark}"),
            &file,
            initial_cost,
            final_cost,
            util::compression_factor(*initial_cost, *final_cost),
            run_time.as_secs_f32(),
            num_libs,
        ))?;
    }

    csv_writer.flush()?;
    Ok(())
}
