"""Check available sentence data files"""
import numpy as np, os

hier = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical'
wiki = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\wikipedia'

if os.path.exists(hier):
    s = np.load(os.path.join(hier, 'sentences.npz'))
    print("Hierarchical:")
    for k in s.keys():
        arr = s[k]
        print("  %s: %s dtype=%s" % (k, str(arr.shape), arr.dtype))
    print("  token_lens max:", s["token_lens"].max())
    print("  word_counts max:", s["word_counts"].max())
    cumsum = np.cumsum(s["word_counts"].astype(np.int64) * 2)
    print("  word_counts cumsum last:", cumsum[-1])

if os.path.exists(wiki):
    print("\nWikipedia:")
    for f in sorted(os.listdir(wiki)):
        fpath = os.path.join(wiki, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        print("  %s: %.1f MB" % (f, size))
        if f.endswith('.npz'):
            w = np.load(fpath)
            for k in w.keys():
                arr = w[k]
                if hasattr(arr, 'shape'):
                    print("    %s: %s dtype=%s" % (k, str(arr.shape), arr.dtype))
                else:
                    print("    %s: scalar=%s" % (k, str(arr)))
else:
    print("\nNo wikipedia directory")
