#!groovy


//loading https://code.sbb.ch/projects/KD_WZU/repos/wzu-pipeline-helper
@Library('python-pipeline-helper') _

library identifier: 'python-helper@master',
        retriever: modernSCM(
                [$class       : 'GitSCMSource',
                 credentialsId: 'fsosebuild',
                 remote       : 'ssh://git@code.sbb.ch:7999/kd_esta_blueprints/esta-python-helper.git'])


pipeline {
    agent { label 'java' }
    tools {
        maven 'Apache Maven 3.3'
        jdk 'Oracle JDK 1.8 64-Bit'
    }
    stages {
        stage('Install & Unit Tests'){
            steps {
                tox_conda_wrapper(
                    JENKINS_CLOSURE: {
                        sh """
# set -e
# set -x
# set -o pipefail

# set up shell for conda
# conda init bash
# source ~/.bashrc

# pip config set global.extra-index-url https://bin.sbb.ch/artifactory/api/pypi/simba.pypi/simple

# conda env create --file environment.yml --force
# conda activate synpop
# pytest --junitxml results.xml
                        """
//                        junit '*.xml'
                        }
                )
            }
        }

        stage('When on develop, do nothing') {
            when {
                branch 'develop'
            }
            steps {
                sh 'ls'
            }
        }

        stage('When on master, Release: Adapt poms, tag, deploy and push.') {
            when {
                branch 'master'
            }
            steps {
                releasePython()
                releasePythonToArtifactory()
            }
        }
    }
}