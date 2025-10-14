# Files

``` bash
resources/
├── bug.hex # hex encoded deployment bytecode of the bug contract
├── bug.ptx # ptx of the bug contract
├── bug.sol # source code of the bug contract
├── bug.tx.hex # hex encoded transaction data triggering the bug: subUnderflow(0, 1)
├── erc20.hex # hex encoded deployment bytecode of the erc20 contract
├── erc20.ptx # ptx of the erc20 contract
├── erc20.sol # source code of the erc20 contract
├── erc20.tx.hex # hex encoded transaction data deploying the erc20 contract
├── librunner.so # the shared library that is used to run the contracts
└── nobug.tx.hex # hex encoded transaction data with no bug: addOverflow(1, 2)
```


# Example execution

(Optional) Create the ptx file in [MAU docker](https://github.com/mao-artifact/mao-artifact):


``` bash
# inside the docker
~/tools/mau/standalone-ptxsema bug.hex -o bug.ll --hex --fsanitize=intsan --dump

llvm-as bug.ll -o bug.bc
llvm-link bug.bc ~/tools/mau/rt.o.bc -o bug-kernel.bc

llc-16 -mcpu=sm_86 bug-kernel.bc -o bug.ptx
```



Test result on RTX 3060:

``` bash
# Execute in docker if you have issues running on the host directly due to the nvidia driver version or the glibc version
# docker run  --gpus all -it -v $(pwd):/app  -w /app augustus/mau-ityfuzz:latest /bin/bash

cargo run --bin mau-run

Loaded shared library: ./resources/librunner.so
CUDA context and module initialized.
GPU memory allocated (d_seeds=0x70dcdaa00000, d_signals=0x70dcda80a600).
setEVMEnv() returned: true
cuDeployTx() returned: true
cuDataCpy() returned: true
cuRunTxs() num transactions = 25600
Executed usdt.hex usdt.tx.hex 25600 transactions. Speed: 0.005567 ms/transaction.
...
...
```


# Run ethtest


```bash
# run in project root folder:
# generate deployment hex file required by ptxsema
python3 json-to-hex.py resources/sdiv.json

# generate PTX from the deployment hex
export MAU_TRACE_PC=1
python3 hex-to-ptx.py resources/sdiv.hex --cleanup

# run the state test in the augustus/mau-ityfuzz:latest docker container
cargo run --bin state_test ./resources/sdiv.json
```

# Comparing traces against go-ethereum

The helper script `scripts/run-trace-comparison.py` expands GeneralStateTests,
produces Mau PTX artifacts, executes both Mau and go-ethereum, and compares the
results. In addition to the program counter sequence, the tool now checks Mau's
first-stack instrumentation (top-of-stack word plus stack depth) whenever both
executors provide that information.

Example:

```bash
python3 scripts/run-trace-comparison.py --ethtest-dir ./GeneralStateTests/VMTests/  --geth-bin goevm --mau-bin target/release/state_test --keep-artifacts --work-dir /tmp/trace-mau --verbose --mau-timeout 10 --goevm-timeout 10
```

Matching cases report `Traces and first stacks match`, and the final summary
includes an extra counter for `Stack mismatches`.
Mismatch reports show the exact step and the differing stack values, e.g.
`Top-of-stack mismatch at step 42: mau=0x..., goevm=0x...`.

Example run excerpt:

```
[627/628] vmTests/swap.json:swap-14-0-0
  Pc Mismatch: PC count mismatch (mau=12, goevm=63)
[628/628] vmTests/swap.json:swap-15-0-0
  Pc Mismatch: PC count mismatch (mau=12, goevm=63)
Summary:
  Total cases: 628, Matches: 0,
  PC mismatches: 438, Trace mismatches: 0,
  Stack mismatches: 0,
  Mau missing: 1, go-ethereum missing: 0, Errors: 89
```

Summary fields:

- `Total cases`: number of fixtures processed (matches + mismatches + errors).
- `Matches`: PC traces agree and (when both sides emit stacks) the depth/top-of-stack do too.
- `PC mismatches`: different PC counts or a divergent PC at some step.
- `Trace mismatches`: same PC count, but values differ (first mismatch shown inline).
- `Stack mismatches`: PC sequences match but top-of-stack words differ at least once.
- `Mau missing` / `go-ethereum missing`: that runner produced no trace.
- `Errors`: script-level failures (build/runtime issues including ptx generation failures) that prevented comparison.
