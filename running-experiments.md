# Running experiments

Running experiments is possible on Lisa, possibly only by one person, but probably by all.  Before running an experiment, make sure to `git pull` and `git commit` any changes.  After the new commit, add a [git tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) that describes the experiment with:

```sh
git tag -a "experiment-<short-name>" [-m "Optionally describe the experiment here."]
```
