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

cargo run

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
