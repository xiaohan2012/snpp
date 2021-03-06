"""
For the utilities
"""
import contexts as ctx
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

from snpp.utils.signed_graph import symmetric_stat, \
    fill_diagonal, \
    make_symmetric, \
    matrix2graph


def test_symmetric_stat(Q1_d):
    c_sym, c_consis = symmetric_stat(Q1_d)
    assert c_sym == 6
    assert c_consis == 4


def test_fill_diagonal():
    N = 2
    m = csr_matrix(np.array([[1, 0], [0, 0]]))
    assert len(set([m[i, i] for i in range(N)])) == 2
    m_new = fill_diagonal(m)
    assert isspmatrix_csr(m_new)
    assert set([m_new[i, i] for i in range(N)]) == {1}


def test_make_symmetric(Q1_d):
    def mask(m):
        """remove inconsistent entries
        """
        inconsis_idx = [(i, j)
                        for i, j in zip(*m.nonzero())
                        if (m[i, j] != 0
                            and m[j, i] != 0
                            and m[j, i] != m[i, j])]
        m_masked = m.copy()
        for i, j in inconsis_idx:
            m_masked[i, j] = m_masked[j, i] = 0
        return m_masked
    Q1_d_masked = mask(Q1_d)
    assert not np.allclose(Q1_d_masked.toarray(), np.transpose(Q1_d_masked.toarray()))
    
    m = make_symmetric(Q1_d)
    assert isspmatrix_csr(m)
    m = m.toarray()
    m_masked = mask(m)
    assert np.allclose(m_masked, np.transpose(m_masked))


def test_matrix2graph(Q1_d):
    gm = matrix2graph(Q1_d, None, multigraph=True)
    g = matrix2graph(Q1_d, None, multigraph=False)
    for i, j in gm.edges():
        s = g[i][j]['sign']
        assert gm[i][j][s]['sign'] == s

    assert gm[0][0][1]['sign'] == g[0][0]['sign'] == 1
    assert gm[2][3][1]['sign'] == g[2][3]['sign'] == 1
    assert gm[0][2][-1]['sign'] == g[0][2]['sign'] == -1
