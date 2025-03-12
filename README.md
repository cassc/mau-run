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



Test execution:

``` bash

cargo run

# Running `bug.hex` with the `nobug.tx.hex` transaction:

...
     Running `target/debug/mau-run`
Loaded shared library: ./resources/librunner.so
CUDA context and module initialized.
GPU memory allocated (d_seeds=0x731c3aa00000, d_signals=0x731c3a80a600).
setEVMEnv() returned: true
cuDeployTx() returned: true
cuDataCpy() returned: true
cuRunTxs() num transactions = 256
postGainCov() returned: true
postGainDu() returned = false
Copied data from GPU: 00000000000000004400000046892d6300000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Storage dump in thread 0
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:0300000000000000000000000000000000000000000000000000000000000000
Storage dump in thread 1
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:0300000000000000000000000000000000000000000000000000000000000040
Storage dump in thread 32
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:0300000000000000000000000000000000000000000000000000000000000080
Cleaned up CUDA context and memory. Exiting.


# Running `bug.hex` with the `bug.tx.hex` transaction:
     Running `target/debug/mau-run`
Loaded shared library: ./resources/librunner.so
CUDA context and module initialized.
GPU memory allocated (d_seeds=0x7c49e6a00000, d_signals=0x7c49e680a600).
setEVMEnv() returned: true
cuDeployTx() returned: true
cuDataCpy() returned: true
cuRunTxs() num transactions = 256
postGainCov() returned: true
[+] RUST-MAU-RUN: Thread#0 found IntegerBug at 30e

postGainDu() returned = true
Copied data from GPU: 000000000000000044000000b01152e200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Storage dump in thread 0
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
Storage dump in thread 1
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff3f
Storage dump in thread 32
------ Slots#1  ------
Key:0000000000000000000000000000000000000000000000000000000000000000
Val:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f
Cleaned up CUDA context and memory. Exiting.
```
