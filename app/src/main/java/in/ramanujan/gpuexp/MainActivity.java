package in.ramanujan.gpuexp;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("gpuexp");
    }

    public native float[] matMulCL(float[] a, float[] b, int n, String kernelSrc);
    public native float[] matMulCPU(float[] a, float[] b, int n);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tvResult = findViewById(R.id.tvResult);
        Button   btnRun   = findViewById(R.id.btnRun);

        btnRun.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                btnRun.setEnabled(false);
                tvResult.setText("Running… (this may take a few seconds)");

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        final String output = runBenchmark();
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tvResult.setText(output);
                                btnRun.setEnabled(true);
                            }
                        });
                    }
                }).start();
            }
        });
    }

    private String runBenchmark() {
        // ── 1. Load kernel ────────────────────────────────────────────────
        String kernelSrc;
        try {
            kernelSrc = loadAsset("matmul.cl");
        } catch (IOException e) {
            return "Error loading kernel: " + e.getMessage();
        }

        // ── 2. Build random matrices (256×256) ────────────────────────────
        int N = 256;
        float[] A = new float[N * N];
        float[] B = new float[N * N];
        Random rng = new Random(42);
        for (int i = 0; i < N * N; i++) {
            A[i] = rng.nextFloat() * 10f;
            B[i] = rng.nextFloat() * 10f;
        }

        // ── 3. CPU run ────────────────────────────────────────────────────
        long cpuStart = System.nanoTime();
        float[] cpuC = matMulCPU(A, B, N);
        long cpuMs = (System.nanoTime() - cpuStart) / 1_000_000;

        // ── 4. GPU run ────────────────────────────────────────────────────
        long gpuStart = System.nanoTime();
        float[] gpuC = matMulCL(A, B, N, kernelSrc);
        long gpuMs = (System.nanoTime() - gpuStart) / 1_000_000;

        // ── 5. Verify correctness (max absolute diff) ─────────────────────
        String verifyStr;
        if (gpuC == null || gpuC.length == 0) {
            return "OpenCL computation failed.\nCheck logcat for details.";
        }
        float maxDiff = 0f;
        for (int i = 0; i < N * N; i++) {
            float diff = Math.abs(gpuC[i] - cpuC[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        verifyStr = String.format("Max |GPU - CPU| diff: %.4f %s",
                maxDiff, maxDiff < 0.1f ? "✓ PASS" : "✗ FAIL");

        // ── 6. Speedup ────────────────────────────────────────────────────
        String speedupStr;
        if (gpuMs > 0) {
            speedupStr = String.format("%.2fx", (double) cpuMs / gpuMs);
        } else {
            speedupStr = "< 1 ms GPU time";
        }

        // ── 7. Sample corner of each result (top-left 4×4) ───────────────
        StringBuilder sb = new StringBuilder();
        sb.append("═══════════════════════════════\n");
        sb.append(String.format(" Matrix size : %d × %d\n", N, N));
        sb.append("═══════════════════════════════\n\n");

        sb.append(String.format(" CPU time    : %d ms\n", cpuMs));
        sb.append(String.format(" GPU time    : %d ms\n", gpuMs));
        sb.append(String.format(" Speedup     : %s\n\n", speedupStr));

        sb.append(" ").append(verifyStr).append("\n\n");

        sb.append("── Top-left 4×4 corner of C ──\n");
        sb.append(" CPU:\n");
        appendCorner(sb, cpuC, N, 4);
        sb.append("\n GPU:\n");
        appendCorner(sb, gpuC, N, 4);

        return sb.toString();
    }

    /** Print the top-left corner×corner submatrix of an N×N row-major array. */
    private static void appendCorner(StringBuilder sb, float[] m, int N, int corner) {
        for (int r = 0; r < corner; r++) {
            sb.append(" [");
            for (int c = 0; c < corner; c++) {
                sb.append(String.format("%8.1f", m[r * N + c]));
                if (c < corner - 1) sb.append(",");
            }
            sb.append(" ]\n");
        }
    }

    private String loadAsset(String name) throws IOException {
        try (InputStream is = getAssets().open(name);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return sb.toString();
        }
    }
}
