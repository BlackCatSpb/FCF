from datasets import load_dataset
import time

t0 = time.time()
ds = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train', streaming=True)
for i, sample in enumerate(ds):
    if i == 0:
        print(f'Title: {sample["title"]}')
        print(f'Text length: {len(sample["text"])} chars')
        print(f'Preview: {sample["text"][:300]}')
    if i >= 3:
        break
print(f'Time: {time.time()-t0:.1f}s')
print(f'OK - datasets works')
