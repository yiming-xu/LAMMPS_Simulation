#!/usr/bin/python
"""
MPI PBS cluster job submission script in Python.

This script runs on a cluster VM, assumes that it can call pbs and runs over a list of jobs.

Required parameters are:
    
    job_names (list): the list of names of the jobs
    modules (2D list/1D list): the list of modules that need to be loaded
    job_commands (2D list/1D list): the list of commands that is to be executed

Optional parameters are:
    source_files (2D list/1D list): the list of files for a simulation
        Default to "$PBS_O_WORKDIR/*", which is the job submission directory
    walltime (list/string): the walltime time of the programme to be run in
        hh:mm:ss format.
        Default to 4:00:00
    proc_nodes (list/int): the number of nodes to be used.
        Default to 1
    proc_cpus (list/int): the number of cpus per nodes to be used.
        Default to 1
    proc_mpiprocs (list/int): the number of mpi processes to be used in total.
        Default to 1
    memory (list/int): the memory in GB used by the simulation.
        Default to 1

Where a single parameter is used instead of a list of parameters (1 per job),
the programme use the same parameter for all jobs.

Alternatively, for convenience, the arguments can be inputted as a dictionary 
expansion. For example:
    def func(arg1,arg2):
        return arg1 + arg2

    args = {"arg1":3,"arg2":4}

    print(func(**args))
    >>> 7

Author:
    Yiming Xu, yiming.xu15@imperial.ac.uk
"""

from subprocess import Popen, PIPE, run
import time
import os


class PBS_Submitter:
    "Defines a class to handling the PBS submission and checking"

    user_path = r"/rds/general/user/yx6015/"
    home_path = r"/rds/general/user/yx6015/home/"
    ephemeral_path = r"/rds/general/user/yx6015/ephemeral/"

    def __init__(self, job_names, job_commands, modules, source_files="$PBS_O_WORKDIR/*", walltime="1:00:00",
                 proc_nodes=1, proc_cpus=1, proc_mpiprocs=1, proc_threads=1, memory=1, **kwargs):

        self.params = {'source_files': source_files,
                       'job_names': job_names,
                       'job_commands': job_commands,
                       'modules': modules,
                       'walltime': walltime,
                       'proc_nodes': proc_nodes,
                       'proc_cpus': proc_cpus,
                       'proc_mpiprocs': proc_mpiprocs,
                       'proc_threads': proc_threads,
                       'memory': memory}

        self.params['job_names'] = job_names

        if type(self.params['job_names']) == list:
            self.no_of_jobs = len(self.params['job_names'])
        else:
            self.no_of_jobs = 1

        # Parameters are checked for length of input. If length < no_of_jobs, they are duplicated
        # until that length as a list.
        for k in self.params.keys():
            # Need to treat the ones that are intrinsically list differently
            if k in ["modules", "job_commands", "source_files"]:
                # If it is not a 2D list of commands (a 1D list for each job)
                if not all([isinstance(x, list) for x in self.params[k]]):
                    self.params[k] = [self.params[k]]*self.no_of_jobs

            else:
                if type(self.params[k]) != list:
                    self.params[k] = [self.params[k]]*self.no_of_jobs
        self.other_args = kwargs

    def run(self):
        "Iterates through and runs all the jobs."
        # Sets TMPDIR environment
        os.environ['TMPDIR'] = r"/rds/general/ephemeral/user/yx6015/ephemeral/"

        pbs_out = []
        pbs_err = []

        for job_no in range(self.no_of_jobs):
            # Loop over your jobs

            # Open a pipe to the qsub command.
            proc = Popen('qsub', shell=True, stdin=PIPE,
                         stdout=PIPE, stderr=PIPE, close_fds=True)

            # Starting PBS Directives
            # proc.stdin.write("#!/bin/bash\n".encode('utf-8'))
            proc.stdin.write(
                "#PBS -N {0}\n".format(self.params['job_names'][job_no]).encode('utf-8'))
            proc.stdin.write(
                "#PBS -o {0}.log\n".format(self.params['job_names'][job_no]).encode('utf-8'))
            proc.stdin.write(
                "#PBS -e {0}.err\n".format(self.params['job_names'][job_no]).encode('utf-8'))

            proc.stdin.write("#PBS -l select={0}:ncpus={1}:mpiprocs={2}:ompthreads={3}:mem={4}gb\n"
                             .format(self.params['proc_nodes'][job_no],
                                     self.params['proc_cpus'][job_no],
                                     self.params['proc_mpiprocs'][job_no],
                                     self.params['proc_threads'][job_no],
                                     self.params['memory'][job_no])
                             .encode('utf-8'))
            proc.stdin.write(
                "#PBS -l walltime={0}\n\n".format(self.params['walltime'][job_no]).encode('utf-8'))

            # Loading modules for the simulation
            for module in self.params['modules'][job_no]:
                proc.stdin.write("module load {0}\n".format(
                    module).encode('utf-8'))

            # Copying input files (*.in) from submission directory to temporary directory for job
            for source_file in self.params['source_files'][job_no]:
                if self.params['proc_nodes'][job_no] == 1:
                    proc.stdin.write("cp {0} $TMPDIR \n".format(source_file).encode('utf-8'))
                else:
                    proc.stdin.write("pbsdsh2 cp {0} $TMPDIR \n".format(source_file).encode('utf-8'))

            # Starting job with mpiexec, it will pick up assigned cores automatically
            for command in self.params['job_commands'][job_no]:
                proc.stdin.write("{0}\n".format(command).encode('utf-8'))

            # Copy output back to directory in $HOME
            proc.stdin.write(
                "mkdir $EPHEMERAL/$PBS_JOBID \n".encode('utf-8'))
            if self.params['proc_nodes'][job_no] == 1:
                proc.stdin.write("cp $TMPDIR/* $EPHEMERAL/$PBS_JOBID/ \n".encode('utf-8'))
            else:
                proc.stdin.write("pbsdsh2 cp $TMPDIR/'*' $EPHEMERAL/$PBS_JOBID/ \n".encode('utf-8'))
            proc.stdin.write(
                "qstat -f $PBS_JOBID \n".encode('utf-8'))
            # Print your job and the system response to the screen as it's submitted
            out, err = proc.communicate()
            proc.kill()

            out = out.decode('utf-8').split()
            if type(out) == list and len(out) > 0:
                pbs_out.append(out[0])
            else:
                pbs_out.append(None)
            pbs_err.append(err)

            if err != b'':
                print("Submitted Job: {0}".format(
                    self.params['job_names'][job_no]))
                print(out, err)

            time.sleep(0.1)

        return pbs_out, pbs_err


def qstat_monitor(jobs_list=None, update_frequency=5):
    "Automatically runs qstat and monitors the output. Requires IPython"
    try:
        from IPython.display import clear_output
    except ImportError:
        print("Warning: clear_output will not work")

    jobs = dict()
    qstat_out_names = ['JobID', 'Job Name',
                       'User', 'Runtime', 'Status', 'Queue']

    while True:
        try:
            clear_output(wait=True)
        except NameError:
            pass

        # Set all to done for each loop
        for k, v in jobs.items():
            v[3] = "Done"

        if jobs_list:
            qstat_out_utf8 = []
            for job in jobs_list:
                time.sleep(0.1)
                qstat_CP = run(["qstat {0}".format(job)],
                               stdout=PIPE, shell=True)
                qstat_out_raw = qstat_CP.stdout.splitlines()
                if len(qstat_out_raw) > 1:
                    qstat_out_utf8.append(qstat_out_raw[2:])
        else:
            qstat_CP = run(["qstat"], stdout=PIPE)
            qstat_out_raw = qstat_CP.stdout.splitlines()
            if len(qstat_out_raw) > 1:
                qstat_out_utf8 = qstat_CP.stdout.splitlines()[2:]

        qstat_out = [x.decode('utf-8') for x in qstat_out_utf8]
        for job in qstat_out:
            splitted_job = job.split()
            # Assign status to dictionary
            jobs[splitted_job[0]] = splitted_job[1:]

        row_format = "{:>16}" * (len(qstat_out_names))
        print(row_format.format(*qstat_out_names))

        for k, v in jobs.items():
            print(row_format.format(k, *v))

        if len(qstat_out_utf8) == 0:
            break

        time.sleep(max(update_frequency/2, 5/2))
        print('Running...')
        time.sleep(max(update_frequency/2, 5/2))
