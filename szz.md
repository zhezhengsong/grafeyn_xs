# Run profile
cargo build --release --manifest-path feynsum-rust/Cargo.toml

mkdir exp_l2
copy the circuits and
mkdir -p scripts raw/static raw/runtime logs
cp ../../exp1-kernel-stats/scripts/rank_l2.py scripts/
ls
echo $BIN

Set the param
'''
    BIN=../feynsum-rust/target/release/feynsum-rust
    export RUST_LOG=error
    ALL=(adder_n28 adder_n433 multiplier_n45 multiplier_n400 bv_n280 qram_n20 qft_n18 ising_n26 knn_n25 dnn_n16)
    RUN=(adder_n28 adder_n433 multiplier_n45 bv_n280 qram_n20 qft_n18 ising_n26 knn_n25 dnn_n16)
    SPARSE=(adder_n28 adder_n433 multiplier_n45 bv_n280 qram_n20)
'''

Print the kernels
```
    for c in "${ALL[@]}"; do $BIN -i circuits/$c.qasm --scheduler gfq-fused --dump-kernels raw/static/$c.jsonl > logs/$c.static.log 2>&1; done
    ls raw/static | wc -l
    head -2 raw/static/dnn_n16.jsonl
```

Run from |0>
```
    for c in "${RUN[@]}"; do echo "orig $c"; $BIN -i circuits/$c.qasm --scheduler gfq-fused --pull-threshold 1.1 --parallelism 4 --kernel-stats raw/runtime/$c.orig.p4.jsonl > logs/$c.orig.p4.log 2>&1 || echo "  FAILED $c"; done
```

Run from 65536 starting points
```
    for c in "${SPARSE[@]}"; do echo "wide $c"; $BIN -i circuits/$c.qasm --scheduler gfq-fused --pull-threshold 1.1 --parallelism 4 --init-random-nonzeros 65536 --seed 1 --kernel-stats raw/runtime/$c.wide.p4.jsonl > logs/$c.wide.p4.log 2>&1 || echo "  FAILED $c"; done
```

python3 scripts/rank_l2.py

python3 scripts/rank_l2_walk.py