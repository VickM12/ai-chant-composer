
from collections import deque

# Load generated chant tokens
with open("generated_structured_chant_tokens.txt", "r") as f:
    tokens = [line.strip() for line in f if line.strip()]

# Parameters
default_velocity = 64
default_off_delay = 5  # number of events later
pending_offs = deque()
result_tokens = []

for i, token in enumerate(tokens):
    result_tokens.append(token)

    # Queue NOTE_OFF injection
    if token.startswith("NOTE_ON_"):
        parts = token.split()
        note = parts[0].split("_")[2]
        delay = default_off_delay
        pending_offs.append((i + delay, note))

    # Inject NOTE_OFFs at the right moment
    while pending_offs and pending_offs[0][0] <= i:
        _, note = pending_offs.popleft()
        note_off = f"NOTE_OFF_{note} VELOCITY_{default_velocity} TIME_10"
        result_tokens.append(note_off)

# Final pass: clear any remaining notes at the end
for _, note in pending_offs:
    result_tokens.append(f"NOTE_OFF_{note} VELOCITY_{default_velocity} TIME_10")

# Save to new file
output_path = "generated_structured_chant_tokens_injected.txt"
with open(output_path, "w") as f:
    for line in result_tokens:
        f.write(line + "\n")

print(f"✅ Injected NOTE_OFF events saved to {output_path}")
