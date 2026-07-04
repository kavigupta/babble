//! The AST defining the boolean-circuit language used by egg-stitch's EPFL
//! circuit benchmark. A circuit is a tree of gates over input leaves: `(and x
//! y)` and `(or x y)` are binary gates, `(not x)` is a unary gate, and `$0`,
//! `$1`, ... are the circuit's input variables.
//!
//! The circuit inputs are modelled as opaque `Input` leaf symbols (not de
//! Bruijn `Var`s): each cone in the corpus is an independent boolean function
//! whose `$N` inputs are free, so they can't be de Bruijn indices (those are
//! reserved for the lambda parameters babble introduces when it abstracts).
//! babble's anti-unification lifts differing inputs into abstraction parameters
//! on its own, exactly as it does for molecule elements.
//!
//! The corpus only uses `and`/`not` gates, but the factoring DSRs introduce
//! `or` (De Morgan / distributivity), so all three gates are part of the
//! language. Modelled on the `molecules` binary, with boolean gates in place of
//! the atom-tree nodes and `Input` leaves in place of element leaves.
//!
//! The DSR file must use plain `=>` rules; expand any bidirectional `<=>` rules
//! into separate forward/backward rules beforehand (egg-stitch's
//! `and_or_demorgan_factor.rewrites` uses `<=>`).

use babble::{
    ast_node::{Arity, AstNode, Expr, Precedence, Printable, Printer},
    learn::{LibId, ParseLibIdError},
    teachable::{BindingExpr, DeBruijnIndex, Teachable},
};
use egg::Symbol;
use std::{
    fmt::{self, Debug, Display, Formatter, Write},
    str::FromStr,
};
use thiserror::Error;

/// The operations/AST nodes of the boolean-circuit language.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Circuit {
    /// Common lambda-calculus / library-learning constructs:
    Var(DeBruijnIndex),
    Lambda,
    LibVar(LibId),
    Lib(LibId),
    Apply,
    List,
    /// The binary AND gate `(and x y)`.
    And,
    /// The binary OR gate `(or x y)` (introduced by the factoring DSRs).
    Or,
    /// The unary NOT gate `(not x)`.
    Not,
    /// A circuit input variable leaf (`$0`, `$1`, ...), kept opaque rather than
    /// as a de Bruijn `Var` since these are free per-cone inputs.
    Input(Symbol),
}

impl Debug for Circuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Arity for Circuit {
    fn min_arity(&self) -> usize {
        match self {
            Self::Var(_) | Self::LibVar(_) | Self::Input(_) => 0,
            Self::Lambda | Self::List | Self::Not => 1,
            Self::Apply | Self::Lib(_) | Self::And | Self::Or => 2,
        }
    }

    fn max_arity(&self) -> Option<usize> {
        match self {
            Self::List => None,
            other => Some(other.min_arity()),
        }
    }
}

impl Display for Circuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(i) => write!(f, "{i}"),
            Self::Lambda => f.write_str("λ"),
            Self::LibVar(ix) => write!(f, "{ix}"),
            Self::Lib(ix) => write!(f, "lib-{ix}"),
            Self::Apply => f.write_str("@"),
            Self::List => f.write_str(":"),
            Self::And => f.write_str("and"),
            Self::Or => f.write_str("or"),
            Self::Not => f.write_str("not"),
            Self::Input(v) => write!(f, "{v}"),
        }
    }
}

/// Failure to parse a token as a [`Circuit`] node.
#[derive(Debug, Error)]
pub(crate) enum ParseCircuitError {
    /// A token that is none of the gates, binders, a `$N` input, or a lib
    /// reference.
    #[error("unknown circuit token {0:?}")]
    Unknown(String),
}

impl FromStr for Circuit {
    type Err = ParseCircuitError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let kind = match s {
            "lambda" | "λ" => Self::Lambda,
            "apply" | "@" => Self::Apply,
            ":" => Self::List,
            "and" => Self::And,
            "or" => Self::Or,
            "not" => Self::Not,
            // `$N` circuit inputs (opaque leaves), then babble's own library
            // symbols (`0`.. LibId / `lib-N`). Anything else is invalid.
            _ if s.starts_with('$') => Self::Input(Symbol::from(s)),
            _ => {
                if let Ok(lv) = s.parse::<LibId>() {
                    Self::LibVar(lv)
                } else if let Ok(lv) = s
                    .strip_prefix("lib-")
                    .ok_or(ParseLibIdError::NoLeadingL)
                    .and_then(str::parse)
                {
                    Self::Lib(lv)
                } else {
                    return Err(ParseCircuitError::Unknown(s.to_owned()));
                }
            }
        };
        Ok(kind)
    }
}

impl Teachable for Circuit {
    fn from_binding_expr<T>(binding_expr: BindingExpr<T>) -> AstNode<Self, T> {
        match binding_expr {
            BindingExpr::Lambda(body) => AstNode::new(Self::Lambda, [body]),
            BindingExpr::Apply(fun, arg) => AstNode::new(Self::Apply, [fun, arg]),
            BindingExpr::Var(index) => AstNode::leaf(Self::Var(index)),
            BindingExpr::LibVar(ix) => AstNode::leaf(Self::LibVar(ix)),
            BindingExpr::Lib(ix, bound_value, body) => {
                AstNode::new(Self::Lib(ix), [bound_value, body])
            }
        }
    }

    fn as_binding_expr<T>(node: &AstNode<Self, T>) -> Option<BindingExpr<&T>> {
        let binding_expr = match node.as_parts() {
            (Self::Lambda, [body]) => BindingExpr::Lambda(body),
            (Self::Apply, [fun, arg]) => BindingExpr::Apply(fun, arg),
            (&Self::Var(index), []) => BindingExpr::Var(index),
            (Self::Lib(ix), [bound_value, body]) => BindingExpr::Lib(*ix, bound_value, body),
            (Self::LibVar(ix), []) => BindingExpr::LibVar(*ix),
            _ => return None,
        };
        Some(binding_expr)
    }

    fn list() -> Self {
        Self::List
    }
}

impl Printable for Circuit {
    fn precedence(&self) -> Precedence {
        match self {
            Self::Var(_) | Self::LibVar(_) | Self::Input(_) => 60,
            Self::List => 50,
            Self::Apply | Self::And | Self::Or | Self::Not => 40,
            Self::Lambda | Self::Lib(_) => 10,
        }
    }

    fn print_naked<W: Write>(expr: &Expr<Self>, printer: &mut Printer<W>) -> fmt::Result {
        match (expr.0.operation(), expr.0.args()) {
            (op @ (Self::And | Self::Or | Self::Not), args) => {
                // Prefix form: `op child child ...` (children parenthesised by
                // precedence as needed).
                write!(printer.writer, "{op}")?;
                for arg in args {
                    printer.writer.write_str(" ")?;
                    printer.print(arg)?;
                }
                Ok(())
            }
            (&Self::List, ts) => {
                let elem = |p: &mut Printer<W>, i: usize| p.print_in_context(&ts[i], 0);
                printer.in_brackets(|p| p.indented(|p| p.vsep(elem, ts.len(), ",")))
            }
            (op, _) => write!(printer.writer, "{op}"),
        }
    }
}
