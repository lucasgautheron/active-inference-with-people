# Running instructions

Note: the following instructions assume that you are using the 'Docker' method for running PsyNet.
If you are instead using the 'virtualenv' method, please refer to the 
[PsyNet documentation website](https://psynetdev.gitlab.io/PsyNet/index.html) for instructions 
for how to run your experiment.

```shell
# Debug the experiment locally
bash docker/psynet debug local  

# Export data from a local experiment
bash docker/psynet export local  

# Run tests
bash docker/run pytest test.py

# Enter a bash terminal (e.g. for debugging)
bash docker/run bash 

# Enter a Python terminal (e.g. for debugging)
bash docker/run python  
```

**Note**: before you run these commands you must have installed and launched
Docker Desktop (see `INSTALL.md`).

There are several commands like this that will soon be fully documented on PsyNet's 
[documentation website](https://psynetdev.gitlab.io/PsyNet).
Please make sure you have followed the instructions in `INSTALL.md` before trying them.

## What happens when I run these commands?

`bash docker/psynet` calls a shell script with the file path `docker/psynet`. 
This shell script does several things:

1. It downloads any required Dallinger/PsyNet images from the internet.
2. It builds a Docker image for your experiment, with reference to `Dockerfile`
   and `requirements.txt`. This includes installing latest versions of any specified
   packages. This step is cached to save time over successive runs.
3. It spins up local Database and Redis services if they are not already available.
4. It launches a Docker container for your experiment.
5. It executes the command `psynet` followed by the arguments that you provided.
6. It filters the console logs to replace certain Docker-specific file paths
   with their equivalents for your local file system. This means that error
   tracebacks will point to the source code that you can edit in your IDE,
   and it means that data exports will link correctly to your computer's 
   export directory.

`bash docker/run` is a more general command that allows you to run any command
directly on the Docker container. For example, running `bash docker/run bash`
allows you to enter an interactive terminal. Note that this command does 
_not_ provide any console log filtering, because this could be quite confusing
for debugging.

## Remote debugging

You can use PyCharm's remote debugger within this Docker-based PsyNet environment.
To set this up, follow these instructions:

1. Click Run > Edit Configurations in PyCharm.
2. Create a new Python Debug Server configuration, and call it something like 'Dockerized Python debug server'.
3. For IDE host name, enter `host.docker.internal`.
4. For port, enter a port of your choice, for example `12345`.

Now start this debug server via your PyCharm interface (this typically involves clicking on a green bug icon).
This should display some code that looks something like this: 

```python
import pydevd_pycharm
pydevd_pycharm.settrace('host.docker.internal', port=12345, stdoutToServer=True, stderrToServer=True)
```

Copy and paste this code into the part of the script that you want to debug, then run it.
If all goes well, the PyCharm interpreter should activate once it reaches this code,
and you can then explore the local state of the program.

## Advanced usage

### Running with local installations of PsyNet and Dallinger (MacOS/Linux only)

This experiment makes heavy use of the Python packages PsyNet and Dallinger.
If you want to debug either of these packages, it is useful to run your 
experiment with local installations of them. The first step is to 
download the source code for these packages and store in them in your 
home directory under their default names as downloaded from source control
(i.e. `~/PsyNet` and `~/Dallinger` respectively). Then you can run your 
experiment as before, but writing `psynet-dev` instead of `psynet, 
for example:

```shell
# Debug the experiment locally with developer installations of 
# PsyNet and Dallinger.
bash docker/psynet-dev debug local 
```

You can change Python code in these packages, save it, then refresh
the browser, and the app should restart with the new code
(note: this hot-refreshing does not yet apply to non-Python assets
such as JavaScript or HTML).

*Warning*: local installations are currently not supported on Windows.

### Running without bash

It is possible to shorten the above command on MacOS and Linux if you first
make the shell scripts in the `docker` folder executable. 
You can do this by running the following command in your working directory:

```shell
chmod +x docker/*
```

You can then invoke the commands like this:

```shell
docker/psynet debug local

docker/psynet export local
```

### Running without Docker

If you are planning to run PsyNet in a local Python installation (i.e. without Docker)
then you can use the same commands but just omitting `bash docker/`, so for example:

```shell
psynet debug local
```
