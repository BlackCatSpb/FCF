"""Check ConceptNet file content — all ASCII, no Unicode in source."""
import re, sys

out_path = 'C:/Users/black/OneDrive/Desktop/FCF/experiments/check_cn_out.txt'
sys.stdout = open(out_path, 'w', encoding='utf-8')

with open('C:/Users/black/OneDrive/Desktop/FCF/real_data/conceptnet/conceptnet_ru.txt', 'rb') as f:
    data = f.read()

print('File size:', len(data))
print('First 100 bytes hex:', data[:100].hex())

# собака in UTF-8
target = bytes([0xd1, 0x81, 0xd0, 0xbe, 0xd0, 0xb1, 0xd0, 0xb0, 0xd0, 0xba, 0xd0, 0xb0])
idx = data.find(target)
if idx >= 0:
    start = max(0, idx - 50)
    end = min(len(data), idx + 200)
    print(f'Found at byte {idx}')
    txt = data[start:end].decode('utf-8', errors='replace')
    print(f'Context: {txt}')
else:
    print('Not found')

# Count form_of lines
form_pattern = b'\xd1\x84\xd0\xbe\xd1\x80\xd0\xbc\xd0\xb0 \xd1\x81\xd0\xbb\xd0\xbe\xd0\xb2\xd0\xb0'
form_count = data.count(form_pattern)
print(f'Total form_of occurrences: {form_count}')

# Check first few lines
lines = data.split(b'\n')
print(f'Total lines: {len(lines)}')
for i in range(3):
    ln = lines[i]
    try:
        txt = ln.decode('utf-8')
        print(f'Line {i}: {txt[:120]}')
    except:
        print(f'Line {i}: bytes {ln[:40].hex()}')

# Check all form_of lines that contain собака-related words
search_words = ['собака', 'слабый', 'комиссия']
for sw in search_words:
    sw_bytes = sw.encode('utf-8')
    count = data.count(sw_bytes)
    print(f'  {sw}: {count} occurrences')
