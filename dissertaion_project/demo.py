import time
import requests
import subprocess
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

# Load Kubernetes configuration
config.load_kube_config()

# Initialize Kubernetes API clients
core_v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()

# Define paths to YAML files
MLFLOW_DEPLOYMENT_YAML = "mlflow-deployment.yaml"
MLFLOW_SERVICE_YAML = "mlflow-service.yaml"
TRAINING_JOB_YAML = "training-deployment.yaml"
INFERENCE_DEPLOYMENT_YAML = "inference-deployment.yaml"
INFERENCE_SERVICE_YAML = "inference-service.yaml"

# Define sample input data for inference
SAMPLE_INPUT = {
    "CustomerID": "CUTTSC18736",
    "CardNumber": "3377890440265668",
    "BankName": "WELLS FARGO BANK NA",
    "CardBrand": "DINERS CLUB INTERNATIONAL",
    "CardType": "CREDIT",
    "IssuingCountry": "UNITED STATES",
    "TransactionAmount": 901.5,
    "TransactionDate": "2025-02-9",
    "Complaint": "Unauthorized Charge",
    "Feedback": "customer not happy with service"
}

def resource_exists(api, namespace, resource_type, resource_name):
    """Check if a Kubernetes resource already exists."""
    try:
        if resource_type == "Service":
            api.read_namespaced_service(resource_name, namespace)
        elif resource_type == "Deployment":
            api.read_namespaced_deployment(resource_name, namespace)
        elif resource_type == "Job":
            api.read_namespaced_job(resource_name, namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise

def apply_yaml(file_path):
    """Apply a Kubernetes YAML file."""
    print(f"Applying {file_path}...")
    try:
        utils.create_from_yaml(
            k8s_client=client.ApiClient(),
            yaml_file=file_path,
            namespace="default"
        )
        print(f"Applied {file_path}")
    except ApiException as e:
        if e.status == 409:  # Conflict (resource already exists)
            print(f"Resource already exists: {file_path}")
        else:
            raise

def wait_for_job_completion(job_name):
    """Wait for the training job to complete."""
    print(f"Waiting for job {job_name} to complete...")
    while True:
        try:
            job_status = batch_v1.read_namespaced_job_status(job_name, namespace="default")
            if job_status.status.succeeded:
                print("Training job completed successfully!")
                break
            elif job_status.status.failed:
                print("Training job failed!")
                break
        except ApiException as e:
            if e.status == 404:
                print(f"Job {job_name} not found. Retrying...")
            else:
                raise
        time.sleep(10)

def wait_for_pod_ready(deployment_name):
    """Wait for the inference pod to be ready."""
    print(f"Waiting for deployment {deployment_name} to be ready...")
    while True:
        pods = core_v1.list_namespaced_pod(namespace="default", label_selector=f"app={deployment_name}")
        if pods.items and all(pod.status.phase == "Running" for pod in pods.items):
            print("Inference pod is ready!")
            break
        time.sleep(10)

def test_inference_api():
    """Test the inference API with sample data."""
    print("Testing inference API...")
    try:
        response = requests.post("http://localhost:8000/predict", json=SAMPLE_INPUT)
        if response.status_code == 200:
            print("Inference API response:", response.json())
        else:
            print("Failed to get a valid response from the inference API.")
    except Exception as e:
        print(f"Error testing inference API: {e}")

def show_mlflow_results():
    """Show MLflow experiment tracking results."""
    print("Showing MLflow experiment tracking results...")
    print("Open the MLflow UI in your browser: http://localhost:5000")
    input("Press Enter to continue after reviewing the MLflow UI...")

def start_port_forwarding(service_name, local_port, target_port):
    """Start port forwarding for a Kubernetes service."""
    print(f"Starting port forwarding for {service_name}...")
    command = f"kubectl port-forward svc/{service_name} {local_port}:{target_port}"
    process = subprocess.Popen(command, shell=True)
    time.sleep(5)  # Wait for port-forwarding to stabilize
    return process

def wait_for_service_ready(service_name):
    """Wait for the service to be ready."""
    print(f"Waiting for service {service_name} to be ready...")
    while True:
        try:
            service_status = core_v1.read_namespaced_service(service_name, namespace="default")
            if service_status.spec.cluster_ip:
                print(f"Service {service_name} is ready!")
                break
        except ApiException as e:
            if e.status == 404:
                print(f"Service {service_name} not found. Retrying...")
            else:
                raise
        time.sleep(10)

def main():
    # Step 1: Deploy the MLflow service
    print("Step 1: Deploying the MLflow service...")
    if not resource_exists(core_v1, "default", "Service", "mlflow-service"):
        apply_yaml(MLFLOW_SERVICE_YAML)
    if not resource_exists(apps_v1, "default", "Deployment", "mlflow-deployment"):  # Ensure this matches the YAML file
        apply_yaml(MLFLOW_DEPLOYMENT_YAML)

    # Wait for MLflow Pod to be ready
    wait_for_pod_ready("mlflow")

    # Start port forwarding for MLflow UI
    mlflow_process = start_port_forwarding("mlflow-service", 5000, 5000)

    # Step 2: Trigger the training job
    print("Step 2: Triggering the training job...")
    if not resource_exists(batch_v1, "default", "Job", "ml-training-job"):  # Ensure this matches the YAML file
        apply_yaml(TRAINING_JOB_YAML)

    # Step 3: Wait for the training job to complete
    wait_for_job_completion("ml-training-job")

    # Step 4: Show MLflow experiment tracking results
    show_mlflow_results()

    # Step 5: Deploy the inference service
    print("Step 5: Deploying the inference service...")
    if not resource_exists(apps_v1, "default", "Deployment", "ml-inference"):
        apply_yaml(INFERENCE_DEPLOYMENT_YAML)
    if not resource_exists(core_v1, "default", "Service", "ml-inference"):
        apply_yaml(INFERENCE_SERVICE_YAML)

    # Wait for inference Pod to be ready
    wait_for_pod_ready("ml-inference")

    # Start port forwarding for inference API
    inference_process = start_port_forwarding("ml-inference", 8000, 8000)

    # Step 6: Wait for the inference pod to be ready
    wait_for_pod_ready("ml-inference")

    # Step 7: Test the inference API
    print("Step 7: Testing the inference API...")
    test_inference_api()

    # Clean up port forwarding processes
    mlflow_process.terminate()
    inference_process.terminate()

if __name__ == "__main__":
    main()