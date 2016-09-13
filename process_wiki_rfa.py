from snpp.utils.data import parse_wiki_rfa, write_wiki_rfa2sp_matrix
from snpp.utils.matrix import load_sparse_csr


df = parse_wiki_rfa('data/wiki-rfa.txt')
write_wiki_rfa2sp_matrix(df, 'data/wiki')
m = load_sparse_csr('data/wiki.npz')
print('#votes and #edges {} {}'.format(df.shape[0], m.nnz))
