https://www.emacswiki.org/emacs/PythonProgrammingInEmacs#toc39

- [Slashdot](http://snap.stanford.edu/data/soc-sign-Slashdot081106.html)
- [Epinion](http://snap.stanford.edu/data/soc-sign-epinions.html)


## pre-processing

- Make the graph into sparse matrix and save to `data/{dataset}.npz`
- Use `snpp.utils.data.load_train_test_graphs` to split/load rain/test data
- For Gephi format, run `python snpp/utils/gephi.py` (remeber to change the `dataset` in the script)