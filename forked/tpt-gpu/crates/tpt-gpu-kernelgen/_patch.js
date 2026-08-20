const fs = require('fs');
const path = String.raw`d:\Programming\1PRODUCTION\Open Source\tpt-gpu\tools\kernel-generator\src\main.rs`;
const original = fs.readFileSync(path, 'utf8');
const oldBlock = "            let mut runner = tptc_rs::bench::BenchRunner::new(config);\r\n            for kernel in [\"vector_add\", \"matmul\", \"softmax\"] {\r\n                let result = runner.run_kernel(kernel, \"f32\", &[1024]);\r\n                println!(\"  {:12}  {:.3} ms  {:.2} GFLOPS\",\r\n                    kernel, result.execution_time_ms, result.throughput_gflops);\r\n            }\r\n        }";
const newBlock = `            println!("  {:12} {:>8} {:>10} {:>8}", "kernel", "time_ms", "GFLOPS", "status");
            println!("  {:12} {:>8} {:>10} {:>8}", "------", "--------", "------", "------");
            let mut runner = tptc_rs::bench::BenchRunner::new(config);
            let kernels: &[(&str, &[i64])] = &[
                ("vector_add", &[1024]),
                ("matmul", &[32, 32]),
                ("softmax", &[1024]),
            ];
            let mut all_pass = true;
            for (kernel, shape) in kernels {
                let result = runner.run_kernel(kernel, "f32", shape);
                let baseline_ms = tptc_rs::bench::baseline_ms(kernel);
                let ratio = if baseline_ms > 0.0 {
                    result.execution_time_ms / baseline_ms
                } else {
                    0.0
                };
                let status = if ratio <= 2.0 {
                    "PASS"
                } else {
                    all_pass = false;
                    "FAIL"
                };
                println!("  {:12} {:>8.3} {:>10.2} {:>8}",
                    kernel, result.execution_time_ms, result.throughput_gflops, status);
            }
            println!();
            if all_pass {
                println!("Quick sanity check PASSED — all kernels within 2x of baseline.");
                println!("Safe to submit.");
            } else {
                println!("Quick sanity check FAILED — one or more kernels exceeded 2x baseline.");
                println!("Investigate before submitting.");
                std::process::exit(1);
            }
        }`;
if (!original.includes(oldBlock)) {
  console.error('ERROR: oldBlock not found. First 200 chars around "let mut runner":');
  const idx = original.indexOf('let mut runner');
  process.stdout.write(JSON.stringify(original.slice(idx, idx + 300)));
  process.exit(2);
}
const updated = original.replace(oldBlock, newBlock);
fs.writeFileSync(path, updated);
console.log(`OK: wrote ${updated.length} chars`);
