pipeline {
    agent {
        label 'docker'
    }

    environment {
        def scmVars = checkout scm
        def commitHash = "${scmVars.GIT_COMMIT.take(7)}"
        def imageName = "hub.kplabs.pl/machine-learning:${commitHash}"
    }

    post {
        failure {
            updateGitlabCommitStatus name: 'build', state: 'failed'
        }
        success {
            updateGitlabCommitStatus name: 'build', state: 'success'
        }
    }
    options {
        gitLabConnection('KPLabs Git')
    }
    triggers {
        gitlab(triggerOnPush: true, triggerOnMergeRequest: true, branchFilterType: 'All')
    }

    stages {
        stage('Build docker') {
            steps {
                sh "docker build . -t ${imageName}"
            }
        }
        stage('Check GPU') {
            steps {
                sh "docker run --gpus all --runtime=nvidia ${imageName} nvidia-smi"
            }
        }
        stage('Unit testing') {
            steps {
                sh "docker run --gpus all --runtime=nvidia ${imageName} pytest -s --durations=100 tests"
            }
        }
        stage('Push docker image to registry') {
            steps {
                sh "docker push ${imageName}"
            }
        }
    }
}
