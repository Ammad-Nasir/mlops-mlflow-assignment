pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Ammad-Nasir/mlops-mlflow-assignment.git'
            }
        }
        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }
        stage('Run Pipeline') {
            steps {
                bat 'python src\\pipeline.py'
            }
        }
    }
}
