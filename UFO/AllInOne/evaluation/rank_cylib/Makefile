all:
	/root/paddlejob/workspace/env_run/anaconda3/envs/py37_torch17_cu11/bin/python setup.py build_ext --inplace
	rm -rf build
clean:
	rm -rf build
	rm -f rank_cy.c *.so
