use std::io::Write;

use monty::{read_into_struct_unchecked, MappedWeights, UnquantisedValueNetwork, ValueNetwork};

fn main() {
    let unquantised: MappedWeights<UnquantisedValueNetwork> =
        unsafe { read_into_struct_unchecked("raw.bin") };

    let quantised = unquantised.data.quantise();

    let mut file = std::fs::File::create("quantised.network").unwrap();

    unsafe {
        let ptr: *const ValueNetwork = quantised.as_ref();
        let slice_ptr: *const u8 = std::mem::transmute(ptr);
        let slice = std::slice::from_raw_parts(slice_ptr, std::mem::size_of::<ValueNetwork>());
        file.write_all(slice).unwrap();
    }
}
