"""
Remote run module

This module enables running tasks in different machines through Jenkins.
"""

import subprocess
import argparse
import json
import sys

from jenkinsapi.jenkins import Jenkins
from jenkinsapi.utils.crumb_requester import CrumbRequester


def process_request(
        host_name: str,
        host_port: str,
        job_name: str,
        credentials_id: str,
        repository_url: str,
        commit_id: str,
        command_prebuild: str,
        command_run: str,
        label: str
) -> None:
    """
    Process a request
    :param host_name: (str) String with the name of the remote host
    :param host_port: (int) Port number of the remote host
    :param job_name: (str) Name of a job to run
    :param credentials_id: (str) Jenkins credentials id to be used during repository fetch
    :param repository_url: (str) Repository URL with code to be run remotely
    :param commit_id: (str) Commit id
    :param command_prebuild: (str) Command to be executed before the docker build
    :param command_run: (str) Command to execute
    :param label: (str) Label of the machine to execute on
    """
    assert host_name is not None, 'Host name is required'
    assert host_port is not None, 'Port number is required'
    assert job_name is not None, 'Job name is required'
    assert credentials_id is not None, 'Credentials ID is required'
    assert repository_url is not None, 'Repository URL is required'
    assert commit_id is not None, 'Commit ID is required'
    assert command_prebuild is not None, 'Prebuild command is required'
    assert command_run is not None, 'Run command is required'
    assert label is not None, 'Label is required'

    if host_port == '80':
        url = 'http://' + host_name
    else:
        url = 'http://' + host_name + ':' + host_port
    server = Jenkins(
        url,
        requester=CrumbRequester(baseurl=url)
    )

    build_params = {
        'commit_id': commit_id,
        'command_prebuild': command_prebuild,
        'command_run': command_run,
        'label': label,
        'credentials_id': credentials_id,
        'repository_url': repository_url,
    }

    job = server[job_name]
    job.invoke(build_params=build_params)
    build = job.get_last_build()
    print("Build queued:")
    print(build.get_result_url())


def get_commit_id() -> str:
    """
    Get id of the latest commit
    :return: (str) Commit id of HEAD
    """

    commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    assert commit_id is not None, 'There was an error when trying to retrieve the latest commit ID'
    return commit_id[:-1].decode(sys.stdout.encoding)


def queue_build(command_prebuild: str,
                command_run: str,
                settings: str='settings.json',
                commit_id = None,
                label: str = 'ml-exp',
                remote_host_name = None,
                remote_host_port = None,
                job_name = None,
                credentials_id = None,
                url = None) -> None:
    """
    Entry point
    """
    configuration = {}
    try:
        with open(settings) as config_file:
            configuration = json.load(config_file)
    except FileNotFoundError:
        print('{} not available'.format(settings))

    commit_id = commit_id or get_commit_id()
    process_request(
        host_name=remote_host_name or configuration.get('remote_host_name', None),
        host_port=remote_host_port or configuration.get('remote_host_port', None),
        job_name=job_name or configuration.get('job_name', None),
        credentials_id=credentials_id or configuration.get('credentials_id', None),
        repository_url=url or configuration.get('url', None),
        commit_id=commit_id,
        command_prebuild=command_prebuild,
        command_run=command_run,
        label=label
    )
