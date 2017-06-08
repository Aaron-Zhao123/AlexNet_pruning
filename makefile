clean:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*.~' -exec rm -rf {} +

# simple cpu run on my local machine
cpurun:
	python run.py

# example: make GPU=0,1 gpurun
# this activates GPU0 and GPU1 for running the thread
gpurun:
	python alex_prune2.py --gpu 3 --data /local/scratch/yaz21/ImageNetData/CLS-LOC
	# CUDA_VISIBLE_DEVICES=$(GPU) python run.py
git-add:
	git add -A
	git commit -m"auto git add all"
	git push
git-commit:
	git commit -am"auto git commit"
	git push
git-merge:
	git fetch
	git merge
