#!/bin/bash
# Compare the local Documents copy of panels_core7 against the Anvil copy,
# per file, to find which files differ. Diagnostic helper.
OUT=/Users/jonathanchang/.claude/jobs/a28206de/tmp
cd /Users/jonathanchang/Documents/ONENAS-paper/panels_core7 || exit 1
find . -type f | sort | while read -r f; do
  printf '%s %s\n' "$(stat -f%z "$f")" "$f"
done > "$OUT/local_sizes.txt"

ssh anvil "cd /anvil/projects/x-cis251123/shared/panels_core7 && find . -type f | sort | while read -r f; do printf '%s %s\n' \"\$(stat -c%s \"\$f\")\" \"\$f\"; done" > "$OUT/anvil_sizes.txt"

echo "local files: $(wc -l < "$OUT/local_sizes.txt")"
echo "anvil files: $(wc -l < "$OUT/anvil_sizes.txt")"
echo "--- files whose sizes differ ---"
diff "$OUT/local_sizes.txt" "$OUT/anvil_sizes.txt" | head -40
