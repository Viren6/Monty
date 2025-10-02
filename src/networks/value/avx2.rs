use super::*;
use std::arch::x86_64::*;

const L2: usize = 16;
const L1S: usize = L1 / 2;

fn ptr<T>(x: &T) -> *const T {
    x
}

fn ptr_mut<T>(x: &mut T) -> *mut T {
    x
}

pub unsafe fn l2(net: &ValueNetwork, l1: &Align64<u8, L1S>) -> Align64<i32, L2> {
    let mut l2 = Align64([0; L2]);

    const CS: usize = 256 / 32;
    const CSB: usize = 256 / 8;
    const L2C: usize = L2 / CS;

    let l1 = &*ptr(l1).cast::<Align64<i32, { L1S / 4 }>>();
    let l2p = ptr_mut(&mut l2).cast::<__m256i>();
    let ws = ptr(&net.l2w).cast::<__m256i>();

    for i in (0..L1S / 4).step_by(2) {
        // QA * QA / (1 << SHIFT)
        let ia = _mm256_set1_epi32(*l1.0.get_unchecked(i));
        let ib = _mm256_set1_epi32(*l1.0.get_unchecked(i + 1));

        for j in 0..L2C {
            // QB
            let this = l2p.add(j);
            let wa = _mm256_load_si256(ws.add(4 * L2 * i / CSB + j));
            let wb = _mm256_load_si256(ws.add(4 * L2 * (i + 1) / CSB + j));

            let sum = _mm256_load_si256(this);
            let p16a = _mm256_maddubs_epi16(ia, wa);
            let p16b = _mm256_maddubs_epi16(ib, wb);
            let p16 = _mm256_add_epi16(p16a, p16b);
            let ones = _mm256_set1_epi16(1);
            let p32 = _mm256_madd_epi16(p16, ones);
            let sum = _mm256_add_epi32(sum, p32);

            // QA * QA * QB / (1 << SHIFT)
            _mm256_store_si256(this, sum);
        }
    }

    l2
}
