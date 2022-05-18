all:
	/root/paddlejob/workspace/env_run/anaconda3/envs/paddle-dev/bin/python setup.py build_ext --inplace
	rm -rf build
clean:
	rm -rf build
	rm -f rank_cy.c *.so
