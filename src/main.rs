#![allow(non_camel_case_types)]

use libloading::{Library, Symbol};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void},
};

// Match the C types:
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUdeviceptr = u64; // typedef in your script

// These structures are not used directly here, but you may need them if your
// real code depends on them in more detail.

fn main() {
    // Adjust the path to your .so as needed
    let lib_path = "./resources/librunner.so";

    // --- 1) Load the shared library dynamically ---
    let lib = unsafe {
        match Library::new(lib_path) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("Failed to load {}: {}", lib_path, e);
                return;
            }
        }
    };
    println!("Loaded shared library: {}", lib_path);

    // --- 2) Define all function pointer types ---

    // extern "C" void InitCudaCtx(CUcontext*, int, CUmodule*, char*);
    type InitCudaCtxFn = unsafe extern "C" fn(*mut CUcontext, c_int, *mut CUmodule, *const c_char);

    // extern "C" void DestroyCuda(CUcontext, CUmodule);
    type DestroyCudaFn = unsafe extern "C" fn(CUcontext, CUmodule);

    // extern "C" void cuMallocAll(CUmodule, CUdeviceptr*, CUdeviceptr*);
    type cuMallocAllFn = unsafe extern "C" fn(CUmodule, *mut CUdeviceptr, *mut CUdeviceptr);

    // extern "C" void cuFreeAll(CUmodule, CUdeviceptr);
    type cuFreeAllFn = unsafe extern "C" fn(CUmodule, CUdeviceptr);

    // extern "C" bool setEVMEnv(CUmodule, unsigned char*, uint64_t, uint64_t);
    type setEVMEnvFn =
        unsafe extern "C" fn(CUmodule, *const c_uchar, c_ulonglong, c_ulonglong) -> bool;

    // extern "C" bool cuDeployTx(CUmodule, uint64_t, unsigned char*, uint32_t);
    type cuDeployTxFn = unsafe extern "C" fn(CUmodule, c_ulonglong, *const c_uchar, c_uint) -> bool;

    // extern "C" bool cuDataCpy(CUdeviceptr, uint64_t, unsigned char*, uint32_t);
    type cuDataCpyFn =
        unsafe extern "C" fn(CUdeviceptr, c_ulonglong, *const c_uchar, c_uint) -> bool;

    // extern "C" uint64_t cuRunTxs(CUmodule, CUdeviceptr, unsigned char*, int32_t);
    type cuRunTxsFn =
        unsafe extern "C" fn(CUmodule, CUdeviceptr, *const c_uchar, c_int) -> c_ulonglong;

    // extern "C" bool postGainCov(CUmodule);
    type postGainCovFn = unsafe extern "C" fn(CUmodule) -> bool;

    // extern "C" void __declspec(dllexport) cuDumpStorage(uint32_t threadId = 0)
    type cuDumpStorageFn = unsafe extern "C" fn(CUmodule, c_uint);

    // extern "C" bool postGainDu(CUdeviceptr dSignals, const char *timeStr)
    type postGainDuFn = unsafe extern "C" fn(CUdeviceptr, *const c_char) -> bool;

    type cuMemcpyDtoH_v2Fn = unsafe extern "C" fn(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        byteCount: usize,
    ) -> c_int;

    // --- 3) Resolve function pointers from the library ---

    fn load_symbol<'lib, F>(lib: &'lib Library, name: &[u8]) -> Symbol<'lib, F> {
        // Return the Symbol itself (function pointer) instead of trying to move out of it
        unsafe {
            lib.get(name).unwrap_or_else(|_| {
                panic!("Could not load symbol: {}", String::from_utf8_lossy(name))
            })
        }
    }

    let init_cuda_ctx: Symbol<InitCudaCtxFn> = load_symbol(&lib, b"InitCudaCtx\0");
    let destroy_cuda: Symbol<DestroyCudaFn> = load_symbol(&lib, b"DestroyCuda\0");
    let cu_malloc_all: Symbol<cuMallocAllFn> = load_symbol(&lib, b"cuMallocAll\0");
    let cu_free_all: Symbol<cuFreeAllFn> = load_symbol(&lib, b"cuFreeAll\0");
    let set_evm_env: Symbol<setEVMEnvFn> = load_symbol(&lib, b"setEVMEnv\0");
    let cu_deploy_tx: Symbol<cuDeployTxFn> = load_symbol(&lib, b"cuDeployTx\0");
    let cu_data_cpy: Symbol<cuDataCpyFn> = load_symbol(&lib, b"cuDataCpy\0");
    let cu_run_txs: Symbol<cuRunTxsFn> = load_symbol(&lib, b"cuRunTxs\0");
    let post_gain_cov: Symbol<postGainCovFn> = load_symbol(&lib, b"postGainCov\0");
    let cu_dump_storage: Symbol<cuDumpStorageFn> =
        load_symbol(&lib, b"_Z11dumpStorageP8CUmod_stm\0");
    let post_gain_du: Symbol<postGainDuFn> = load_symbol(&lib, b"postGainDu\0");
    let cu_memcpy_dtoh: Symbol<cuMemcpyDtoH_v2Fn> = load_symbol(&lib, b"cuMemcpyDtoH_v2\0");

    // --- 4) Prepare variables for calls ---
    let mut cu_ctx: CUcontext = std::ptr::null_mut();
    let mut cu_module: CUmodule = std::ptr::null_mut();
    let gpu_id: c_int = 0;

    let tests = vec![
        ("usdt.ptx", "usdt.hex", "usdt.tx.hex"),
        ("bug.ptx", "bug.hex", "bug.tx.hex"),
        ("bug.ptx", "bug.hex", "nobug.tx.hex"),
    ];

    // run tests
    for (kernel_file, deploy_file, tx_file) in tests.into_iter() {
        // Adjust your kernel file path if necessary
        let kernel_path = format!("./resources/{kernel_file}");

        // Convert kernel_path to a C string
        let kernel_cstring = match CString::new(kernel_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to build CString from kernel path: {}", e);
                return;
            }
        };

        // --- 5) Call InitCudaCtx(...) ---
        unsafe {
            init_cuda_ctx(&mut cu_ctx, gpu_id, &mut cu_module, kernel_cstring.as_ptr());
        }
        println!("CUDA context and module initialized.");

        // --- 6) Allocate GPU memory with cuMallocAll ---
        let mut d_seeds: CUdeviceptr = 0;
        let mut d_signals: CUdeviceptr = 0;
        unsafe {
            cu_malloc_all(cu_module, &mut d_seeds, &mut d_signals);
        }
        println!(
            "GPU memory allocated (d_seeds={:#x}, d_signals={:#x}).",
            d_seeds, d_signals
        );

        // --- 7) setEVMEnv ---
        let sender_address = hex::decode("acbf3a12181192fcebef19e27292c98eff62cc76").unwrap();
        let timestamp: c_ulonglong = 123456789;
        let blocknum: c_ulonglong = 1;
        let env_ok =
            unsafe { set_evm_env(cu_module, sender_address.as_ptr(), timestamp, blocknum) };
        println!("setEVMEnv() returned: {}", env_ok);

        // --- 8) cuDeployTx ---
        // Read the deployment data (0x...) from a file
        let tx_data_deploy = std::fs::read(format!("./resources/{deploy_file}")).unwrap();

        // and decode as bytes
        let tx_data_deploy = hex::decode(tx_data_deploy).expect("Decode tx data as hex failed");

        let tx_value: c_ulonglong = 100;
        let tx_size = tx_data_deploy.len() as c_uint;
        let deploy_ok =
            unsafe { cu_deploy_tx(cu_module, tx_value, tx_data_deploy.as_ptr(), tx_size) };
        println!("cuDeployTx() returned: {}", deploy_ok);

        // --- 9) Copy transaction data to GPU with cuDataCpy ---
        // Python snippet used a hex string:
        let tx_bytes = hex::decode(
            std::fs::read(format!("./resources/{tx_file}")).expect("Read tx data failed"),
        )
        .expect("Decode tx data as hex failed");
        let tx_size2 = tx_bytes.len() as c_uint;
        // Let's define tx_value again as 0 for the copy
        let tx_value2: c_ulonglong = 0;

        let copy_ok = unsafe { cu_data_cpy(d_seeds, tx_value2, tx_bytes.as_ptr(), tx_size2) };
        println!("cuDataCpy() returned: {}", copy_ok);

        // --- 10) Run transaction with cuRunTxs ---
        let arg_type_map = [0x68_u8; 1];

        let mut total_executions = 0;
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let executions: c_ulonglong = unsafe {
                cu_run_txs(
                    cu_module,
                    d_seeds,
                    arg_type_map.as_ptr(),
                    arg_type_map.len() as c_int,
                )
            };

            if executions == 0 {
                eprintln!("cuRunTxs() returned 0 executions. Exiting.");
                break;
            }

            total_executions += executions;
        }

        let elapsed = start.elapsed();
        println!("cuRunTxs() num transactions = {}", total_executions);

        // Print average time
        let avg_time = elapsed.as_micros() as f64 / total_executions as f64 / 1000.0;
        println!(
            "Executed {} {} {} transactions. Speed: {:.6} ms/transaction.",
            deploy_file, tx_file, total_executions, avg_time
        );

        // --- 11) postGainCov ---
        let cov_ok = unsafe { post_gain_cov(cu_module) };
        println!("postGainCov() returned: {}", cov_ok);

        let msg = CString::new("RUST-MAU-RUN: ").unwrap();
        let du_gained = unsafe { post_gain_du(d_signals, msg.as_ptr()) };
        println!("postGainDu() returned = {}", du_gained);

        let mut host_buf = vec![0u8; 128];

        let res = unsafe {
            cu_memcpy_dtoh(
                host_buf.as_mut_ptr() as *mut c_void,
                d_seeds,
                host_buf.len(),
            )
        };

        if res != 0 {
            eprintln!("cuMemcpyDtoH_v2 returned error code {}", res);
        } else {
            println!("Copied data from GPU: {}", hex::encode(&host_buf[..]));
        }

        unsafe {
            vec![0, 10, 233, 255].into_iter().for_each(|i| {
                println!("Storage dump in thread {}", i);
                cu_dump_storage(cu_module, i);
            });
        }
        // --- 12) Cleanup: free GPU memory and destroy context ---
        unsafe {
            cu_free_all(cu_module, d_seeds);
            destroy_cuda(cu_ctx, cu_module);
        }
        println!("Cleaned up CUDA context and memory. Exiting.");
    }
}
