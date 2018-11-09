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
                 proc_nodes=1, proc_cpus=1, proc_mpiprocs=1, memory=1, **kwargs):

        self.params = {'source_files': source_files,
                       'job_names': job_names,
                       'job_commands': job_commands,
                       'modules': modules,
                       'walltime': walltime,
                       'proc_nodes': proc_nodes,
                       'proc_cpus': proc_cpus,
                       'proc_mpiprocs': proc_mpiprocs,
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
                # If it is not a 2D list of commands for all jobs (must be correct length and not identical)
                if not (len(self.params[k]) == self.no_of_jobs and self.params[k][1:] != self.params[k][:-1]):
                    self.params[k] = [self.params[k]]*self.no_of_jobs

            else:
                if type(self.params[k]) != list:
                    self.params[k] = [self.params[k]]*self.no_of_jobs

    def run(self):
        "Iterates through and runs all the jobs."
        pbs_out = []
        pbs_err = []
        
        for job_no in range(self.no_of_jobs):
        # Loop over your jobs

            # Open a pipe to the qsub command.
            proc = Popen('qsub', shell=True, stdin=PIPE,
                         stdout=PIPE, stderr=PIPE, close_fds=True)

            # Starting PBS Directives
            proc.stdin.write("#!/bin/bash\n".encode('utf-8'))
            proc.stdin.write(
                "#PBS -N {0}\n".format(self.params['job_names'][job_no]).encode('utf-8'))
            proc.stdin.write(
                "#PBS -o {0}.log\n".format(self.params['job_names'][job_no]).encode('utf-8'))
            proc.stdin.write(
                "#PBS -e {0}.err\n".format(self.params['job_names'][job_no]).encode('utf-8'))

            proc.stdin.write("#PBS -l select={0}:ncpus={1}:mpiprocs={2}:mem={3}gb\n"
                             .format(self.params['proc_nodes'][job_no], 
                                     self.params['proc_cpus'][job_no], 
                                     self.params['proc_mpiprocs'][job_no], 
                                     self.params['memory'][job_no])
                             .encode('utf-8'))
            proc.stdin.write(
                "#PBS -l walltime={0}\n\n".format(self.params['walltime'][job_no]).encode('utf-8'))

            # Loading modules for the simulation
            if type(self.params['modules'][job_no]) == list:
                for module in self.params['modules'][job_no]:
                    proc.stdin.write("module load {0}\n".format(module).encode('utf-8'))
            else:
                proc.stdin.write("module load {0}\n".format(self.params['modules'][job_no]).encode('utf-8'))

            # Copying input files (*.in) from submission directory to temporary directory for job
            if type(self.params['source_files'][job_no]) == list:
                for source_file in self.params['source_files'][job_no]:
                    proc.stdin.write("cp {0} . \n".format(source_file).encode('utf-8'))
            else:
                proc.stdin.write("cp {0} . \n".format(self.params['source_files'][job_no]).encode('utf-8'))

            # Starting job with mpiexec, it will pick up assigned cores automatically
            if type(self.params['job_commands'][job_no]) == list:
                for command in self.params['job_commands'][job_no]:
                    proc.stdin.write("{0}\n".format(command).encode('utf-8'))
            else:
                proc.stdin.write("{0}\n".format(self.params['job_commands'][job_no]).encode('utf-8'))

            # Copy output back to directory in $HOME
            proc.stdin.write(
                "mkdir $HOME/cx1_out/$PBS_JOBID \n".encode('utf-8'))
            proc.stdin.write(
                "cp * $HOME/cx1_out/$PBS_JOBID/ \n".encode('utf-8'))

            # Print your job and the system response to the screen as it's submitted
            out, err = proc.communicate()
            proc.kill()

            out = out.decode('utf-8').split()
            if type(out) == list:
                pbs_out.append(out[0])
            else:
                pbs_out.append(None)
            pbs_err.append(err)
            
            if err != b'':
                print("Submitted Job: {0}".format(self.params['job_names'][job_no]))
                print(out, err)

            time.sleep(0.1)

        return pbs_out, pbs_err

def qstat_monitor(update_frequency=5, jobs_list = None):
    "Automatically runs qstat and monitors the output. Requires IPython"
    try:
        from IPython.display import clear_output
    except ImportError:
        print("Warning: clear_output will not work")
    
    jobs = dict()
    qstat_out_names = ['JobID', 'Job Name', 'User', 'Runtime', 'Status', 'Queue']
    
    while True:
        try:
            clear_output(wait=True)
        except NameError:
            pass
        
        # Set all to done
        for k, v in jobs.items():
            v[3] = "Done"
        
        qstat_CP = run(["qstat"], stdout=PIPE)
        qstat_out_utf8 = qstat_CP.stdout.splitlines()[2:]

        qstat_out = [x.decode('utf-8') for x in qstat_out_utf8]
        for job in qstat_out:
            splitted_job = job.split()
            jobs[splitted_job[0]] = splitted_job[1:]
        
        row_format ="{:>16}" * (len(qstat_out_names))
        print(row_format.format(*qstat_out_names))

        for k, v in jobs.items():
            if k in jobs_list:
                print(row_format.format(k, *v))
            
        if len(qstat_out_utf8) == 0:
            break
            
        time.sleep(max(update_frequency/2, 5/2))
        print('Running...')
        time.sleep(max(update_frequency/2, 5/2))