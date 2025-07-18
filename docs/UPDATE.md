# Updating

To update your experiment to use a new PsyNet version, 
simply specify an updated version in requirements.txt.
This version will be automatically installed and used next time you run
a Docker command such as `docker/psynet` or `docker/run`.

PsyNet automatically generates various helper scripts and documentation files
such as this one. To generate updated resources, navigate to your experiment directory
and run the following Docker command:

```shell
bash docker/psynet update-scripts
```

or simply

```shell
psynet update-scripts
```

if you are not using Docker.
