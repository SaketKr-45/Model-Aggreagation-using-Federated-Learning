🛡️ Model Aggregation using Federated LearningA simulation of a secure, privacy-preserving machine learning system using Federated Learning (FL). This project demonstrates how multiple clients can collaboratively train a central model (for fraud detection) using the Flower (flwr) framework and PyTorch, without ever sharing their local data.🌟 Key FeaturesFederated Learning Server: A central server (server.py) that orchestrates the training process.Multiple Clients: Simulates three independent clients (new_client1.py, new_client2.py, new_client3.py), each with its own local dataset.Data Privacy: Clients train locally; only model weights (not raw data) are sent to the server.Custom Aggregation: The server uses a custom SaveModelFedAvg strategy to save the aggregated global model to the saved_models/ directory after every round.Local Data Preprocessing: Each client includes a custom PyTorch Dataset to load, clean, and prepare its local CSV data for training.Automated Runner: A simple script (run.py) to launch the server and all clients simultaneously in separate terminals (for Windows).💡 How it WorksThe system follows a classic federated learning "hub-and-spoke" architecture.ShutterstockServer Starts: The server.py script starts and listens for clients. It's configured to wait for 3 clients (min_available_clients=3).Clients Connect: The three new_client.py scripts are launched. Each one loads its own local datasets/clientN_dataset.csv.Training Rounds: The server begins orchestrating 15 rounds of training:a. Distribution: The server sends the current global model weights to all clients.b. Local Training: Each client trains the received model on its own local data for one epoch.c. Aggregation: Clients send their updated model weights (not their data) back to the server.d. Model Update: The server (using FedAvg) averages the weights from all clients to create an improved global model.e. Save Model: The server's custom SaveModelFedAvg strategy saves the newly aggregated global model as a .pkl file in the saved_models/ directory.Completion: After 15 rounds, the process is complete, and the saved_models/ directory contains the model's state from each round.📁 Project StructureModel-Aggreagation-using-Federated-Learning/
├── datasets/
│   ├── client1_dataset.csv   # (Data for client 1)
│   ├── client2_dataset.csv   # (Data for client 2)
│   └── client3_dataset.csv   # (Data for client 3)
├── saved_models/
│   ├── global_model_round_1.pkl
│   └── (...more models...)
│
├── server.py                 # The FL Server
├── new_client1.py            # Client 1 (with data preprocessing)
├── new_client2.py            # Client 2 (with data preprocessing)
├── new_client3.py            # Client 3 (with data preprocessing)
├── run.py                    # Script to run the simulation (for Windows)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
🚀 Getting Started1. PrerequisitesPython (tested with 3.11.5)pip (Python package installer)2. InstallationClone the repository:git clone [https://github.com/your-username/Model-Aggreagation-using-Federated-Learning.git](https://github.com/your-username/Model-Aggreagation-using-Federated-Learning.git)
cd Model-Aggreagation-using-Federated-Learning
Create and activate a virtual environment (Recommended):# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
Install the required packages:pip install -r requirements.txt
3. Prepare Your DataThis simulation is designed to run on your own data.Create the datasets folder if it doesn't exist.Place your three client data files inside it with the exact names the scripts expect:datasets/client1_dataset.csvdatasets/client2_dataset.csvdatasets/client3_dataset.csv⚡️ Running the SimulationYou must start the server first, and then the three clients.Option 1: Automated (Windows Only)The run.py script automates this process by opening four new terminal windows.python run.py
Option 2: Manual (All Platforms)Open four separate terminals and run the following commands, one in each terminal.Terminal 1 (Server):python server.py
Terminal 2 (Client 1):python new_client1.py
Terminal 3 (Client 2):python new_client2.py
Terminal 4 (Client 3):python new_client3.py
The server will begin the training process as soon as all three clients have successfully connected. You will see output in the server terminal as it saves the model from each round.

