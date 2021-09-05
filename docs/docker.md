# Docker

Instead of installing the body composition code directly on your system, you
can use the Dockerfile in the repository to build a docker image that already
has all the code and dependencies built in. If you are unfamiliar with docker,
I suggest reading this [guide](https://docs.docker.com/get-started/) and
familiarizing yourself with docker before continuing.

### Building the Docker Image

After you have cloned the body composition repository, change directory to the
root of the cloned repository (where the file called `Dockerfile` is found) and
run:

```bash
$ sudo docker build -t body_comp:latest .
```

The process of building the image will take several minutes to complete.


### Running the Container

Once built, you should start an interactive session within a container in order
to run the code. Remember that you will need to mount in the locations of any
data you want to access from inside the container using the `-v` option, as
well as anywhere you want to write the results to. Writing the results within
the container's filesystem means they will be deleted when the container is
shut down.

For example, the following command will run the container with data on your
system (`/path/to/data`) mounted into the container so that it's visible at
`/data`:

```bash
$ sudo docker run -it -v /path/to/data/:/data/ body_comp:latest bash
```

### Running Code Inside the Container

After executing the command above, you should find yourself in the
`/bin/body_comp` directory in the container. You will find all the files for
running the model in this directory (as described in the other documentation
pages). The environment should all be set up and you should be able to run
these python files straight away.
