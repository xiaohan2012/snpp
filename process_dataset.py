from snpp.utils.matrix import save_sparse_csr, load_sparse_csr
from snpp.utils.data import load_csv_as_sparse
from snpp.utils.signed_graph import symmetric_stat

dataset = 'slashdot'
# dataset = 'epinions'
csv_path = 'data/soc-sign-{}.txt'.format(dataset)
npz_mat_path = 'data/{}'.format(dataset)
raw_mat_path = npz_mat_path + '.npz'


def save_raw_matrix():
    m = load_csv_as_sparse(csv_path)
    save_sparse_csr(npz_mat_path, m)


def print_symmetric_stat(m):
    print('symmetric_stat...')
    c1, c2 = symmetric_stat(m)
    print('symmetric ratio = {} / {} = {}'.format(c1, m.nnz, c1 / m.nnz))
    print('consistent ratio = {} / {} = {}'.format(c2, c1, c2 / c1))
    

def main():
    print('loading {}..'.format(dataset))
    m = load_sparse_csr(raw_mat_path)



if __name__ == '__main__':
    # save_raw_matrix()
    main()
    
