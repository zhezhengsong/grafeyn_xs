use bit_vec::BitVec;
use std::fmt::{self, Display, Formatter};
use std::hash::Hash;
use std::sync::RwLock;

use super::{AtomicBasisIdx, BasisIdx};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BasisIdxUnlimited {
    bits: BitVec,
}

impl Display for BasisIdxUnlimited {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let n = self.bits.len();
        for i in 0..n {
            if self.bits[n - i - 1] {
                write!(f, "1")?;
            } else {
                write!(f, "0")?;
            }
        }
        Ok(())
    }
}

impl BasisIdx for BasisIdxUnlimited {
    fn get(&self, qi: usize) -> bool {
        self.bits.get(qi).unwrap_or(false)
    }

    fn flip(&self, qi: usize) -> Self {
        let mut new_bits = self.bits.clone();
        resize_if_needed(&mut new_bits, qi);
        new_bits.set(qi, !self.bits.get(qi).unwrap_or(false));
        Self { bits: new_bits }
    }

    fn zeros() -> Self {
        Self {
            bits: BitVec::new(),
        }
    }

    fn set(&self, qi: usize) -> Self {
        let mut new_bits = self.bits.clone();
        resize_if_needed(&mut new_bits, qi);
        new_bits.set(qi, true);
        Self { bits: new_bits }
    }

    fn unset(&self, qi: usize) -> Self {
        let mut new_bits = self.bits.clone();
        resize_if_needed(&mut new_bits, qi);
        new_bits.set(qi, false);
        Self { bits: new_bits }
    }

    fn swap(&self, qi1: usize, qi2: usize) -> Self {
        let mut new_bits = self.bits.clone();
        resize_if_needed(&mut new_bits, qi1);
        resize_if_needed(&mut new_bits, qi2);

        // read through `get`: bits past the end of a short vector are 0
        // (indexing `self.bits` directly panicked on e.g. `zeros().swap(3, 5)`)
        new_bits.set(qi1, self.get(qi2));
        new_bits.set(qi2, self.get(qi1));

        Self { bits: new_bits }
    }

    fn from_idx(_idx: usize) -> Self {
        unimplemented!("push/pull dense should never be called")
    }

    fn as_idx(&self) -> usize {
        unimplemented!("push/pull dense should never be called")
    }

    fn empty_key(num_qubits: usize) -> Self {
        let mut bits = BitVec::from_elem(num_qubits + 1, false);
        bits.set(num_qubits, true);
        Self { bits }
    }

    fn as_bytes(&self) -> Vec<u8> {
        self.bits.to_bytes()
    }
}

fn resize_if_needed(bits: &mut BitVec, qi: usize) {
    if qi >= bits.len() {
        bits.grow(qi - bits.len() + 1, false);
    }
}

impl AtomicBasisIdx<BasisIdxUnlimited> for RwLock<BasisIdxUnlimited> {
    fn empty_key(num_qubits: usize) -> Self {
        RwLock::new(BasisIdxUnlimited::empty_key(num_qubits))
    }

    fn load(&self) -> BasisIdxUnlimited {
        self.read().unwrap().clone()
    }

    fn compare_exchange(
        &self,
        current: BasisIdxUnlimited,
        new: BasisIdxUnlimited,
    ) -> Result<BasisIdxUnlimited, ()> {
        let mut guard = self.write().unwrap();

        if *guard == current {
            *guard = new;
            Ok(current)
        } else {
            Err(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_beyond_length() {
        let b = BasisIdxUnlimited::zeros().set(5);
        let s = b.swap(5, 70);
        assert!(!s.get(5) && s.get(70));
        assert!(!BasisIdxUnlimited::zeros().swap(3, 9).get(3));
    }
}
