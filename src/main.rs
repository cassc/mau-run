#![allow(non_camel_case_types)]

use libloading::{Library, Symbol};
use std::{
    ffi::CString,
    os::raw::{c_char, c_int, c_uchar, c_uint, c_ulonglong},
    time::Instant,
};

const TRANSACTION_SIZE: usize = 0x800;
const SEED_HEADER_LEN: usize = 12;
const MAX_CALLDATA_LEN: usize = TRANSACTION_SIZE - SEED_HEADER_LEN;

type InitCudaCtxFn = unsafe extern "C" fn(c_int, *const c_char);
type DestroyCudaFn = unsafe extern "C" fn();
type CuMallocAllFn = unsafe extern "C" fn();
type CuFreeAllFn = unsafe extern "C" fn();
type SetEvmEnvFn =
    unsafe extern "C" fn(*const c_uchar, *const c_uchar, *const c_uchar) -> bool;
type CuDeployTxFn = unsafe extern "C" fn(c_ulonglong, *const c_uchar, c_uint) -> bool;
type CuExtSeedsFn = unsafe extern "C" fn(*const c_uchar, c_uint);
type CuRunTxsFn = unsafe extern "C" fn(*const c_uchar, c_uint) -> c_ulonglong;
type GetCudaExecResFn = unsafe extern "C" fn(*mut c_ulonglong, *mut c_ulonglong) -> bool;
type CuDumpStorageFn = unsafe extern "C" fn(c_uint);
type CuVersionFn = unsafe extern "C" fn() -> c_uint;
fn load_symbol<'lib, F>(lib: &'lib Library, name: &[u8]) -> Symbol<'lib, F> {
    unsafe {
        lib.get(name)
            .unwrap_or_else(|_| panic!("Could not load symbol: {}", String::from_utf8_lossy(name)))
    }
}

fn pack_u256(value: u64) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[24..].copy_from_slice(&value.to_be_bytes());
    buf
}

fn build_seed(callvalue: u64, calldata: &[u8]) -> Vec<u8> {
    if calldata.len() > MAX_CALLDATA_LEN {
        panic!(
            "calldata length {} exceeds supported limit {}",
            calldata.len(),
            MAX_CALLDATA_LEN
        );
    }

    let mut seed = vec![0u8; SEED_HEADER_LEN + calldata.len()];
    seed[..8].copy_from_slice(&callvalue.to_le_bytes());
    seed[8..12].copy_from_slice(&(calldata.len() as u32).to_le_bytes());
    seed[SEED_HEADER_LEN..].copy_from_slice(calldata);
    seed
}

fn main() {
    let lib_path = "./resources/librunner.so";
    let lib = unsafe {
        match Library::new(lib_path) {
            Ok(lib) => lib,
            Err(err) => {
                eprintln!("Failed to load {}: {}", lib_path, err);
                return;
            }
        }
    };
    println!("Loaded shared library: {}", lib_path);

    let init_cuda_ctx: Symbol<InitCudaCtxFn> = load_symbol(&lib, b"InitCudaCtx\0");
    let destroy_cuda: Symbol<DestroyCudaFn> = load_symbol(&lib, b"DestroyCuda\0");
    let cu_malloc_all: Symbol<CuMallocAllFn> = load_symbol(&lib, b"cuMallocAll\0");
    let cu_free_all: Symbol<CuFreeAllFn> = load_symbol(&lib, b"cuFreeAll\0");
    let set_evm_env: Symbol<SetEvmEnvFn> = load_symbol(&lib, b"setEVMEnv\0");
    let cu_deploy_tx: Symbol<CuDeployTxFn> = load_symbol(&lib, b"cuDeployTx\0");
    let cu_ext_seeds: Symbol<CuExtSeedsFn> = load_symbol(&lib, b"cuExtSeeds\0");
    let cu_run_txs: Symbol<CuRunTxsFn> = load_symbol(&lib, b"cuRunTxs\0");
    let get_cuda_exec_res: Symbol<GetCudaExecResFn> =
        load_symbol(&lib, b"getCudaExecRes\0");
    let cu_dump_storage: Symbol<CuDumpStorageFn> = load_symbol(&lib, b"cuDumpStorage\0");
    let cu_version: Symbol<CuVersionFn> = load_symbol(&lib, b"cuVersion\0");

    let version = unsafe { cu_version() };
    println!("Runner version reported: {}", version);

    let gpu_id: c_int = 0;
    let tests = vec![
        ("usdt.ptx", "usdt.hex", "usdt.tx.hex"),
        ("bug.ptx", "bug.hex", "bug.tx.hex"),
        ("bug.ptx", "bug.hex", "nobug.tx.hex"),
    ];

    for (kernel_file, deploy_file, tx_file) in tests {
        let kernel_path = format!("./resources/{kernel_file}");
        let kernel_cstring = match CString::new(kernel_path.clone()) {
            Ok(cstr) => cstr,
            Err(err) => {
                eprintln!("Failed to create CString for {}: {}", kernel_path, err);
                break;
            }
        };

        unsafe {
            init_cuda_ctx(gpu_id, kernel_cstring.as_ptr());
        }
        println!("CUDA context initialized for {}", kernel_path);

        unsafe {
            cu_malloc_all();
        }
        println!("GPU buffers allocated.");

        let self_address =
            hex::decode("acbf3a12181192fcebef19e27292c98eff62cc76").expect("decode addr failed");
        if self_address.len() != 20 {
            eprintln!("Contract address must be 20 bytes, got {}", self_address.len());
            break;
        }

        let timestamp_bytes = pack_u256(123_456_789);
        let blocknum_bytes = pack_u256(1);
        let env_ok = unsafe {
            set_evm_env(
                self_address.as_ptr(),
                timestamp_bytes.as_ptr(),
                blocknum_bytes.as_ptr(),
            )
        };
        println!("setEVMEnv() returned: {}", env_ok);

        let deploy_bytes = hex::decode(
            std::fs::read(format!("./resources/{deploy_file}")).expect("read deploy hex failed"),
        )
        .expect("decode deploy hex failed");
        let deploy_ok =
            unsafe { cu_deploy_tx(100, deploy_bytes.as_ptr(), deploy_bytes.len() as c_uint) };
        println!("cuDeployTx() returned: {}", deploy_ok);

        let tx_bytes = hex::decode(
            std::fs::read(format!("./resources/{tx_file}")).expect("read tx hex failed"),
        )
        .expect("decode tx hex failed");
        let seed_buffer = build_seed(0, &tx_bytes);

        unsafe {
            cu_ext_seeds(seed_buffer.as_ptr(), seed_buffer.len() as c_uint);
        }
        println!("Loaded {} bytes of calldata into seeds.", tx_bytes.len());

        let arg_type_map = [0x68_u8; 1];
        let mut total_executions = 0_u64;
        let start = Instant::now();
        for _ in 0..100 {
            let executed =
                unsafe { cu_run_txs(arg_type_map.as_ptr(), arg_type_map.len() as c_uint) };
            if executed == 0 {
                eprintln!("cuRunTxs() returned 0 executions. Stopping loop.");
                break;
            }
            total_executions += executed as u64;
        }

        if total_executions > 0 {
            let elapsed = start.elapsed();
            let avg_ms = elapsed.as_secs_f64() * 1000.0 / total_executions as f64;
            println!(
                "Executed {} {} {} transactions. Speed: {:.6} ms/transaction.",
                deploy_file, tx_file, total_executions, avg_ms
            );
        } else {
            println!("No executions reported for {} {}", deploy_file, tx_file);
        }

        let mut coverage = 0_u64;
        let mut bug_sig = 0_u64;
        let has_signal = unsafe { get_cuda_exec_res(&mut coverage, &mut bug_sig) };
        println!(
            "getCudaExecRes() => has_signal={}, coverage={}, bug_sig={}",
            has_signal, coverage, bug_sig
        );

        unsafe {
            cu_dump_storage(0);
        }

        unsafe {
            cu_free_all();
            destroy_cuda();
        }
        println!("Cleaned up CUDA context for {}", kernel_file);
    }

}
