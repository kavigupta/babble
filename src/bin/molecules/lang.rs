//! The AST defining the molecule language used by egg-stitch's molecule
//! benchmark. A molecule is an atom tree: each atom node is `(<bond><degree> E
//! n1 n2 ...)` where `<bond>` is the bond to the parent (`m` = root/none, `s`
//! single, `d` double, `t` triple), `<degree>` is the neighbour count
//! (including the parent), `E` is the element symbol, and the remaining slots
//! are the child neighbours. So a root `mN` has arity `N+1` (element + N
//! children) and a bonded `sN`/`dN`/`tN` has arity `N` (element + N-1 children,
//! the Nth neighbour being the parent).

use babble::{
    ast_node::{Arity, AstNode, Expr, Precedence, Printable, Printer},
    learn::{LibId, ParseLibIdError},
    teachable::{BindingExpr, DeBruijnIndex, Teachable},
};
use egg::Symbol;
use std::{
    fmt::{self, Debug, Display, Formatter, Write},
    num::ParseIntError,
    str::FromStr,
};

/// The bond order linking an atom to its parent (`m` marks the root, which has
/// no parent bond).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Bond {
    Root,
    Single,
    Double,
    Triple,
}

impl Bond {
    /// The single-character prefix this bond uses in the surface syntax.
    fn prefix(self) -> char {
        match self {
            Self::Root => 'm',
            Self::Single => 's',
            Self::Double => 'd',
            Self::Triple => 't',
        }
    }

    /// Parse a bond prefix character, or `None` if it isn't one.
    fn from_prefix(c: char) -> Option<Self> {
        Some(match c {
            'm' => Self::Root,
            's' => Self::Single,
            'd' => Self::Double,
            't' => Self::Triple,
            _ => return None,
        })
    }
}

/// Parse a `<bond><degree>` atom head (e.g. `s4`, `m1`), or `None` if `s` is
/// not one (a leading bond-prefix char followed by a non-empty decimal degree).
fn try_parse_atom(s: &str) -> Option<Molecule> {
    let mut chars = s.chars();
    let bond = Bond::from_prefix(chars.next()?)?;
    let rest = chars.as_str();
    if rest.is_empty() {
        return None;
    }
    rest.parse::<u8>().ok().map(|degree| Molecule::Atom(bond, degree))
}

/// The operations/AST nodes of the molecule language.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Molecule {
    /// Common lambda-calculus / library-learning constructs:
    Var(DeBruijnIndex),
    Lambda,
    LibVar(LibId),
    Lib(LibId),
    Apply,
    List,
    /// An atom node `(<bond><degree> E ...)`; `degree` is the neighbour count.
    Atom(Bond, u8),
    /// An element symbol leaf (`C`, `H`, `O`, ...).
    Element(Symbol),
}

impl Debug for Molecule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Arity for Molecule {
    fn min_arity(&self) -> usize {
        match self {
            Self::Var(_) | Self::LibVar(_) | Self::Element(_) => 0,
            Self::Lambda | Self::List => 1,
            Self::Apply | Self::Lib(_) => 2,
            // Root atoms have no parent, so all `degree` neighbours are
            // children (plus the element slot); bonded atoms reserve one
            // neighbour for the parent.
            Self::Atom(Bond::Root, degree) => *degree as usize + 1,
            Self::Atom(_, degree) => *degree as usize,
        }
    }

    fn max_arity(&self) -> Option<usize> {
        match self {
            Self::List => None,
            other => Some(other.min_arity()),
        }
    }
}

impl Display for Molecule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(i) => write!(f, "{i}"),
            Self::Lambda => f.write_str("λ"),
            Self::LibVar(ix) => write!(f, "{ix}"),
            Self::Lib(ix) => write!(f, "lib-{ix}"),
            Self::Apply => f.write_str("@"),
            Self::List => f.write_str(":"),
            Self::Atom(bond, degree) => write!(f, "{}{degree}", bond.prefix()),
            Self::Element(e) => write!(f, "{e}"),
        }
    }
}

impl FromStr for Molecule {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let kind = match s {
            "lambda" | "λ" => Self::Lambda,
            "apply" | "@" => Self::Apply,
            ":" => Self::List,
            // `<bond><degree>`: a bond-prefix char followed by a decimal degree.
            _ if try_parse_atom(s).is_some() => try_parse_atom(s).expect("checked above"),
            _ => {
                if let Ok(index) = s.parse::<DeBruijnIndex>() {
                    Self::Var(index)
                } else if let Ok(lv) = s.parse::<LibId>() {
                    Self::LibVar(lv)
                } else if let Ok(lv) = s
                    .strip_prefix("lib-")
                    .ok_or(ParseLibIdError::NoLeadingL)
                    .and_then(str::parse)
                {
                    Self::Lib(lv)
                } else {
                    // Anything else is an element symbol leaf.
                    Self::Element(Symbol::from(s))
                }
            }
        };
        Ok(kind)
    }
}

impl Teachable for Molecule {
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

impl Printable for Molecule {
    fn precedence(&self) -> Precedence {
        match self {
            Self::Var(_) | Self::LibVar(_) | Self::Element(_) => 60,
            Self::List => 50,
            Self::Apply | Self::Atom(..) => 40,
            Self::Lambda | Self::Lib(_) => 10,
        }
    }

    fn print_naked<W: Write>(expr: &Expr<Self>, printer: &mut Printer<W>) -> fmt::Result {
        match (expr.0.operation(), expr.0.args()) {
            (&Self::Element(e), []) => write!(printer.writer, "{e}"),
            (op @ Self::Atom(..), args) => {
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
