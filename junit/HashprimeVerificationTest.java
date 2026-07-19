import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

/**
 * Drives a model-produced {@code hashprime} program across multiple N checkpoints
 * from the A000040 manifest and asserts the SHA-256 hex digest it prints to
 * stdout matches the manifest digest.
 *
 * Configured via system properties (passed on the java command line):
 *   hashprime.dir  - directory containing hashprime.class (working dir).
 *   manifest.file  - path to the A000040 manifest JSON. Default: ./A000040.json
 *   max.n          - largest N to test (checkpoints above this are skipped).
 *                    Default: 100000000 (delays beyond this become extreme).
 *
 * The class under test MUST be named "hashprime" (lowercase) and accept a single
 * integer argument N on the command line, printing ONLY the lowercase-hex
 * SHA-256 of the prime bytes (2..N, each followed by '\n') to stdout.
 */
public class HashprimeVerificationTest {

    private static final String CLASS_NAME = "hashprime";

    private Path hashprimeDir() {
        String d = System.getProperty("hashprime.dir");
        return (d == null || d.isEmpty()) ? Path.of(".") : Path.of(d).toAbsolutePath();
    }

    private Path manifestFile() {
        String m = System.getProperty("manifest.file");
        return (m == null || m.isEmpty()) ? Path.of("A000040.json") : Path.of(m).toAbsolutePath();
    }

    private long maxN() {
        String v = System.getProperty("max.n");
        try {
            return v == null ? 100_000_000L : Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            return 100_000_000L;
        }
    }

    /** Load checkpoint N -> expected SHA-256 hex from the manifest. */
    private Map<Long, String> loadCheckpoints() throws IOException {
        String text = Files.readString(manifestFile());
        // Minimal parse: manifest has "checkpoint_hashes": { "N": { "hash": "..." }, ... }
        Map<Long, String> out = new LinkedHashMap<>();
        int idx = text.indexOf("\"checkpoint_hashes\"");
        if (idx < 0) {
            throw new IOException("manifest missing checkpoint_hashes");
        }
        String rest = text.substring(idx);
        // Walk brace-balanced to find the checkpoint_hashes object.
        int start = rest.indexOf('{', rest.indexOf('"')); // first '{' after the key
        int depth = 0, end = -1;
        for (int i = start; i < rest.length(); i++) {
            char c = rest.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') { depth--; if (depth == 0) { end = i; break; } }
        }
        String obj = rest.substring(start, end + 1);
        // Extract "N": { "hash": "HEX" }
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
                "\"(\\d+)\"\\s*:\\s*\\{\\s*\"hash\"\\s*:\\s*\"([0-9a-fA-F]+)\"");
        java.util.regex.Matcher matcher = p.matcher(obj);
        while (matcher.find()) {
            out.put(Long.parseLong(matcher.group(1)), matcher.group(2).toLowerCase());
        }
        return out;
    }

    /** Run `java hashprime N` in hashprimeDir, capture the SHA-256 hex digest it
     *  prints to stdout (ONLY the hash, per the benchmark contract), return it. */
    private String runAndHash(long n, Path dir) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("java", "-cp", dir.toString(), CLASS_NAME, Long.toString(n));
        pb.directory(dir.toFile());
        pb.redirectErrorStream(false);
        Process proc = pb.start();
        // Capture stdout (the printed hash) and stderr for diagnostics.
        String out;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
            out = br.lines().reduce((a, b) -> a + "\n" + b).orElse("");
        }
        String err;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(proc.getErrorStream()))) {
            err = br.lines().reduce((a, b) -> a + "\n" + b).orElse("");
        }
        boolean finished = proc.waitFor(10, java.util.concurrent.TimeUnit.MINUTES);
        if (!finished) {
            proc.destroyForcibly();
            throw new IOException("hashprime timed out at N=" + n + (err.isEmpty() ? "" : "\n" + err));
        }
        if (proc.exitValue() != 0) {
            throw new IOException("hashprime exited " + proc.exitValue() + " at N=" + n
                    + (err.isEmpty() ? "" : "\n" + err));
        }
        // The program prints ONLY the lowercase-hex SHA-256 of the prime bytes.
        String actual = out.strip().toLowerCase();
        if (actual.isEmpty()) {
            throw new IOException("hashprime printed no hash to stdout at N=" + n);
        }
        return actual;
    }

    @TestFactory
    Stream<DynamicTest> verifyAllCheckpoints() throws IOException {
        Path dir = hashprimeDir();
        long cap = maxN();
        Map<Long, String> checkpoints = loadCheckpoints();
        List<Long> ns = new ArrayList<>();
        for (long n : checkpoints.keySet()) {
            if (n <= cap) ns.add(n);
        }
        ns.sort(Long::compareTo);

        List<DynamicTest> tests = new ArrayList<>();
        for (long n : ns) {
            String expected = checkpoints.get(n);
            tests.add(dynamicTest("N=" + n, () -> {
                String actual = runAndHash(n, dir);
                assertEquals(expected, actual,
                        () -> "SHA-256 printed to stdout at N=" + n
                                + " did not match manifest (expected " + expected + ", got " + actual + ")");
            }));
        }
        assertTrue(!tests.isEmpty(), "no checkpoints <= max.n found in manifest");
        return tests.stream();
    }
}
