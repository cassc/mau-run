#![allow(non_camel_case_types)]

use libloading::{Library, Symbol};
use serde_json::Value;
use std::{
    convert::TryInto,
    env,
    ffi::CString,
    fs,
    os::raw::{c_char, c_int, c_uchar, c_uint, c_ulonglong},
    path::Path,
    ptr,
    time::Instant,
};

const GPU_THREADS_FALLBACK: usize = 64 * 16;

type InitCudaCtxFn = unsafe extern "C" fn(c_int, *const c_char);
type DestroyCudaFn = unsafe extern "C" fn();
type CuMallocAllFn = unsafe extern "C" fn();
type CuFreeAllFn = unsafe extern "C" fn();
type SetEvmEnvFn =
    unsafe extern "C" fn(*const c_uchar, *const c_uchar, *const c_uchar) -> bool;
type CuDeployTxFn = unsafe extern "C" fn(c_ulonglong, *const c_uchar, c_uint) -> bool;
type CuRunTxsFn = unsafe extern "C" fn(*const c_uchar, c_uint) -> c_ulonglong;
type GetCudaExecResFn = unsafe extern "C" fn(*mut c_ulonglong, *mut c_ulonglong) -> bool;
type CuDumpStorageFn = unsafe extern "C" fn(c_uint);
type CuAddCallerPoolFn = unsafe extern "C" fn(*const c_uchar, c_uint);
type CuAddAddressPoolFn = unsafe extern "C" fn(*const c_uchar, c_uint);
type CuLoadStorageFn = unsafe extern "C" fn(*const c_uchar, c_uint, c_uint);
type CuSetStorageMapFn = unsafe extern "C" fn(c_uint, c_uint);
type CuLoadSeedFn = unsafe extern "C" fn(
    *const c_uchar,
    *const c_uchar,
    *const c_uchar,
    c_uint,
    c_uint,
    c_uint,
);
type CuGetThreadsFn = unsafe extern "C" fn() -> c_uint;

fn load_symbol<'lib, F>(lib: &'lib Library, name: &[u8]) -> Symbol<'lib, F> {
    unsafe {
        lib.get(name)
            .unwrap_or_else(|_| panic!("Could not load symbol: {}", String::from_utf8_lossy(name)))
    }
}

fn decode_hex_bytes(input: &str) -> Vec<u8> {
    let trimmed = input.trim_start_matches("0x");
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.len() % 2 == 0 {
        hex::decode(trimmed).expect("failed to decode hex string")
    } else {
        let mut padded = String::with_capacity(trimmed.len() + 1);
        padded.push('0');
        padded.push_str(trimmed);
        hex::decode(padded).expect("failed to decode padded hex string")
    }
}

fn decode_hex_fixed<const N: usize>(input: &str) -> [u8; N] {
    let bytes = decode_hex_bytes(input);
    if bytes.len() > N {
        panic!(
            "decoded hex length {} exceeds target size {} for {}",
            bytes.len(),
            N,
            input
        );
    }
    let mut out = [0u8; N];
    let offset = N - bytes.len();
    out[offset..].copy_from_slice(&bytes);
    out
}

fn normalize_hex_id(input: &str) -> String {
    input.trim_start_matches("0x").to_ascii_lowercase()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Slot {
    key: [u64; 4],
    val: [u64; 4],
}

struct AccountState {
    state_id: u32,
    normalized: String,
    address: [u8; 20],
    storage_slots: Vec<Slot>,
}

fn make_slot(key_hex: &str, value_hex: &str) -> Slot {
    let key_bytes = decode_hex_fixed::<32>(key_hex);
    let value_bytes = decode_hex_fixed::<32>(value_hex);
    let mut slot = Slot {
        key: [0; 4],
        val: [0; 4],
    };
    for i in 0..4 {
        let key_slice: [u8; 8] = key_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        let val_slice: [u8; 8] = value_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
        slot.key[i] = u64::from_be_bytes(key_slice);
        slot.val[i] = u64::from_be_bytes(val_slice);
    }
    slot
}

fn load_bytes_from_hex_file(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        return None;
    }
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {}", path, err));
    let cleaned = contents.trim();
    Some(
        hex::decode(cleaned.trim_start_matches("0x"))
            .unwrap_or_else(|err| panic!("Failed to decode {}: {}", path, err)),
    )
}

fn parse_state_test(json_path: &str) -> (String, Value) {
    let json_text =
        fs::read_to_string(json_path).unwrap_or_else(|err| panic!("Failed to read {}: {}", json_path, err));
    let document: Value =
        serde_json::from_str(&json_text).unwrap_or_else(|err| panic!("Failed to parse {}: {}", json_path, err));
    let test_object = document
        .as_object()
        .and_then(|obj| obj.iter().next())
        .unwrap_or_else(|| panic!("State test {} did not contain any entries", json_path));
    (test_object.0.clone(), test_object.1.clone())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let json_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/app/resources/expPower2.json");
    let kernel_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("./resources/expPower2.ptx");
    let deploy_hex_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("./resources/expPower2.hex");

    if !Path::new(json_path).exists() {
        eprintln!("JSON file {} not found; aborting.", json_path);
        return;
    }
    if !Path::new(kernel_path).exists() {
        eprintln!("Kernel PTX {} not found; aborting.", kernel_path);
        return;
    }

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
    let cu_run_txs: Symbol<CuRunTxsFn> = load_symbol(&lib, b"cuRunTxs\0");
    let get_cuda_exec_res: Symbol<GetCudaExecResFn> =
        load_symbol(&lib, b"getCudaExecRes\0");
    let cu_dump_storage: Symbol<CuDumpStorageFn> = load_symbol(&lib, b"cuDumpStorage\0");
    let cu_add_caller_pool: Symbol<CuAddCallerPoolFn> =
        load_symbol(&lib, b"cuAddCallerPool\0");
    let cu_add_address_pool: Symbol<CuAddAddressPoolFn> =
        load_symbol(&lib, b"cuAddAddressPool\0");
    let cu_load_storage: Symbol<CuLoadStorageFn> =
        load_symbol(&lib, b"cuLoadStorage\0");
    let cu_set_storage_map: Symbol<CuSetStorageMapFn> =
        load_symbol(&lib, b"cuSetStorageMap\0");
    let cu_load_seed: Symbol<CuLoadSeedFn> = load_symbol(&lib, b"cuLoadSeed\0");

    let gpu_threads = unsafe {
        match lib.get::<CuGetThreadsFn>(b"cuGetThreads\0") {
            Ok(sym) => {
                let count = sym() as usize;
                println!("cuGetThreads() reported {} GPU threads.", count);
                count
            }
            Err(_) => {
                println!(
                    "cuGetThreads symbol not exported; defaulting to {} GPU threads.",
                    GPU_THREADS_FALLBACK
                );
                GPU_THREADS_FALLBACK
            }
        }
    };

    let (test_name, test_body) = parse_state_test(json_path);
    println!("Running state test '{}' via GPU runner", test_name);

    let env_obj = test_body
        .get("env")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("State test '{}' missing env section", test_name));
    let pre_obj = test_body
        .get("pre")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("State test '{}' missing pre state", test_name));
    let tx_obj = test_body
        .get("transaction")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("State test '{}' missing transaction", test_name));

    let contract_to = tx_obj
        .get("to")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("Transaction in '{}' missing 'to'", test_name));
    let sender = tx_obj
        .get("sender")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("Transaction in '{}' missing 'sender'", test_name));
    let callvalue_hex = tx_obj
        .get("value")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
        .unwrap_or("0x0");
    let calldata_hex = tx_obj
        .get("data")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
        .unwrap_or("0x");
    let timestamp_hex = env_obj
        .get("currentTimestamp")
        .and_then(Value::as_str)
        .unwrap_or("0x0");
    let block_hex = env_obj
        .get("currentNumber")
        .and_then(Value::as_str)
        .unwrap_or("0x0");

    let contract_bytes = decode_hex_fixed::<20>(contract_to);
    let sender_bytes = decode_hex_fixed::<20>(sender);
    let callvalue_bytes = decode_hex_fixed::<32>(callvalue_hex);
    let timestamp_bytes = decode_hex_fixed::<32>(timestamp_hex);
    let block_bytes = decode_hex_fixed::<32>(block_hex);
    let calldata_bytes = decode_hex_bytes(calldata_hex);

    let mut accounts = Vec::new();
    for (index, (addr, account)) in pre_obj.iter().enumerate() {
        let normalized = normalize_hex_id(addr);
        let address_bytes = decode_hex_fixed::<20>(addr);
        let storage_slots = account
            .get("storage")
            .and_then(Value::as_object)
            .map(|storage_obj| {
                storage_obj
                    .iter()
                    .map(|(slot_key, slot_val)| {
                        let value_str = slot_val
                            .as_str()
                            .unwrap_or_else(|| panic!("Storage value for {} missing string", slot_key));
                        make_slot(slot_key, value_str)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        accounts.push(AccountState {
            state_id: index as u32,
            normalized,
            address: address_bytes,
            storage_slots,
        });
    }

    let target_norm = normalize_hex_id(contract_to);
    let contract_state_id = accounts
        .iter()
        .find(|acct| acct.normalized == target_norm)
        .map(|acct| acct.state_id)
        .unwrap_or(0);

    let kernel_cstring = CString::new(kernel_path).expect("Failed to convert kernel path to CString");

    unsafe {
        init_cuda_ctx(0, kernel_cstring.as_ptr());
    }
    println!("CUDA context initialized for {}", kernel_path);

    unsafe {
        cu_malloc_all();
    }
    println!("GPU buffers allocated.");

    let mut address_pool_len = 0u32;
    for account in &accounts {
        address_pool_len += 1;
        unsafe {
            cu_add_address_pool(account.address.as_ptr(), address_pool_len);
        }
    }

    let mut caller_pool_len = 0u32;
    caller_pool_len += 1;
    unsafe {
        cu_add_caller_pool(sender_bytes.as_ptr(), caller_pool_len);
    }

    let env_ok = unsafe {
        set_evm_env(
            contract_bytes.as_ptr(),
            timestamp_bytes.as_ptr(),
            block_bytes.as_ptr(),
        )
    };
    println!("setEVMEnv() returned: {}", env_ok);

    for account in &accounts {
        unsafe {
            cu_set_storage_map(account.state_id, account.state_id);
        }
        if !account.storage_slots.is_empty() {
            unsafe {
                cu_load_storage(
                    account.storage_slots.as_ptr() as *const c_uchar,
                    account.storage_slots.len() as c_uint,
                    account.state_id,
                );
            }
        }
    }

    if let Some(deploy_bytes) = load_bytes_from_hex_file(deploy_hex_path) {
        let deploy_ok = unsafe {
            cu_deploy_tx(
                0,
                deploy_bytes.as_ptr(),
                deploy_bytes.len() as c_uint,
            )
        };
        println!("cuDeployTx() returned: {}", deploy_ok);
    } else {
        println!(
            "Constructor hex {} not found; assuming contract already deployed.",
            deploy_hex_path
        );
    }

    let data_ptr = if calldata_bytes.is_empty() {
        ptr::null()
    } else {
        calldata_bytes.as_ptr()
    };
    for thread_id in 0..gpu_threads {
        unsafe {
            cu_load_seed(
                sender_bytes.as_ptr(),
                callvalue_bytes.as_ptr(),
                data_ptr,
                calldata_bytes.len() as c_uint,
                contract_state_id,
                thread_id as c_uint,
            );
        }
    }

    let start = Instant::now();
    let arg_type_map = [0x68_u8; 1];
    let executed =
        unsafe { cu_run_txs(arg_type_map.as_ptr(), arg_type_map.len() as c_uint) };
    if executed == 0 {
        eprintln!("cuRunTxs() returned 0 for state test '{}'", test_name);
    } else {
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / executed as f64;
        println!(
            "State test '{}' executed {} threads. Avg {:.6} ms/thread.",
            test_name, executed, avg_ms
        );
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
    println!("State test '{}' finished and context cleaned up.", test_name);
}
