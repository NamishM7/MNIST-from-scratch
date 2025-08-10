{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f00da5ca",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:16.132386Z",
     "iopub.status.busy": "2025-08-10T15:30:16.132079Z",
     "iopub.status.idle": "2025-08-10T15:30:18.086785Z",
     "shell.execute_reply": "2025-08-10T15:30:18.085567Z"
    },
    "papermill": {
     "duration": 1.961466,
     "end_time": "2025-08-10T15:30:18.088812",
     "exception": false,
     "start_time": "2025-08-10T15:30:16.127346",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/digit-recognizer/sample_submission.csv\n",
      "/kaggle/input/digit-recognizer/train.csv\n",
      "/kaggle/input/digit-recognizer/test.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a135df36",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:18.096824Z",
     "iopub.status.busy": "2025-08-10T15:30:18.095361Z",
     "iopub.status.idle": "2025-08-10T15:30:18.101268Z",
     "shell.execute_reply": "2025-08-10T15:30:18.100344Z"
    },
    "papermill": {
     "duration": 0.011043,
     "end_time": "2025-08-10T15:30:18.102935",
     "exception": false,
     "start_time": "2025-08-10T15:30:18.091892",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "38807422",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:18.109959Z",
     "iopub.status.busy": "2025-08-10T15:30:18.109413Z",
     "iopub.status.idle": "2025-08-10T15:30:21.894279Z",
     "shell.execute_reply": "2025-08-10T15:30:21.893132Z"
    },
    "papermill": {
     "duration": 3.790843,
     "end_time": "2025-08-10T15:30:21.896670",
     "exception": false,
     "start_time": "2025-08-10T15:30:18.105827",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8b9609b2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:21.903279Z",
     "iopub.status.busy": "2025-08-10T15:30:21.902962Z",
     "iopub.status.idle": "2025-08-10T15:30:22.705829Z",
     "shell.execute_reply": "2025-08-10T15:30:22.704754Z"
    },
    "papermill": {
     "duration": 0.808672,
     "end_time": "2025-08-10T15:30:22.708068",
     "exception": false,
     "start_time": "2025-08-10T15:30:21.899396",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = np.array(data)\n",
    "m, n = data.shape\n",
    "np.random.shuffle(data)\n",
    "\n",
    "data_dev = data[0:1000].T\n",
    "Y_dev = data_dev[0]\n",
    "X_dev = data_dev[1:n]\n",
    "X_dev = X_dev / 255\n",
    "\n",
    "data_train = data[1000:m].T\n",
    "Y_train = data_train[0]\n",
    "X_train = data_train[1:n]\n",
    "X_train = X_train / 255\n",
    "_,m_train = X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "718df386",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:22.715580Z",
     "iopub.status.busy": "2025-08-10T15:30:22.715225Z",
     "iopub.status.idle": "2025-08-10T15:30:22.728375Z",
     "shell.execute_reply": "2025-08-10T15:30:22.727265Z"
    },
    "papermill": {
     "duration": 0.019209,
     "end_time": "2025-08-10T15:30:22.730194",
     "exception": false,
     "start_time": "2025-08-10T15:30:22.710985",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def init_params():\n",
    "    w1 = np.random.rand(10, 784) - 0.5\n",
    "    b1 = np.random.rand(10, 1) - 0.5\n",
    "    w2 = np.random.rand(10, 10) - 0.5\n",
    "    b2 = np.random.rand(10, 1) - 0.5\n",
    "\n",
    "    return w1, b1, w2, b2\n",
    "\n",
    "def ReLU(z):\n",
    "    return np.maximum(0, z)\n",
    "\n",
    "def softmax(z):\n",
    "    A = np.exp(z) / sum(np.exp(z))\n",
    "    return A\n",
    "    \n",
    "def forward_prop(w1, b1, w2, b2, X):\n",
    "    z1 = w1.dot(X) + b1\n",
    "    A1 = ReLU(z1)\n",
    "    z2 = w2.dot(A1) + b2\n",
    "    A2 = softmax(z2)\n",
    "    return z1, A1, z2, A2\n",
    "\n",
    "def one_hot(Y):\n",
    "    one_hot_Y = np.zeros((Y.size, Y.max() + 1))\n",
    "    one_hot_Y[np.arange(Y.size), Y] = 1\n",
    "    one_hot_Y = one_hot_Y.T\n",
    "    return one_hot_Y\n",
    "\n",
    "def deriv_relu(z):\n",
    "    return z > 0\n",
    "    \n",
    "def back_prop(z1, A1, z2, A2, w2, X, Y):\n",
    "    \n",
    "    one_hot_Y = one_hot(Y)\n",
    "    dZ2 = A2 - one_hot_Y\n",
    "    dW2 = 1 / m * dZ2.dot(A1.T)\n",
    "    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)\n",
    "\n",
    "    dZ1 = w2.T.dot(dZ2) * deriv_relu(z1)\n",
    "    dW1 = 1 / m * dZ1.dot(X.T)\n",
    "    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)\n",
    "    return dW1, db1, dW2, db2\n",
    "\n",
    "def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):\n",
    "    w1 = w1 - alpha * dW1\n",
    "    b1 = b1 - alpha * db1\n",
    "    w2 = w2 - alpha * dW2\n",
    "    b2 = b2 - alpha * db2\n",
    "    return w1, b1, w2, b2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "247912a6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:22.736737Z",
     "iopub.status.busy": "2025-08-10T15:30:22.736396Z",
     "iopub.status.idle": "2025-08-10T15:30:22.743579Z",
     "shell.execute_reply": "2025-08-10T15:30:22.742642Z"
    },
    "papermill": {
     "duration": 0.01221,
     "end_time": "2025-08-10T15:30:22.745305",
     "exception": false,
     "start_time": "2025-08-10T15:30:22.733095",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def get_predictions(A2):\n",
    "    return np.argmax(A2, 0)\n",
    "\n",
    "def get_accuracy(predictions, Y):\n",
    "    print(predictions, Y)\n",
    "    return np.sum(predictions == Y) / Y.size\n",
    "\n",
    "def gradient_descent(X, Y, iterations, alpha):\n",
    "    w1, b1, w2, b2 = init_params()\n",
    "    for i in range(iterations):\n",
    "        z1, A1, z2, A2 = forward_prop(w1, b1, w2, b2, X)\n",
    "        dW1, db1, dW2, db2 = back_prop(z1, A1, z2, A2, w2, X, Y)\n",
    "        w1, b1, w2, b2, = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)\n",
    "        if i % 50 == 0:\n",
    "            print(\"Iteration: \", i)\n",
    "            print(\"Accuracy: \", get_accuracy(get_predictions(A2), Y))\n",
    "    return w1, b1, w2, b2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8c0896ad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-10T15:30:22.751743Z",
     "iopub.status.busy": "2025-08-10T15:30:22.751461Z",
     "iopub.status.idle": "2025-08-10T15:31:08.106652Z",
     "shell.execute_reply": "2025-08-10T15:31:08.105537Z"
    },
    "papermill": {
     "duration": 45.360996,
     "end_time": "2025-08-10T15:31:08.109013",
     "exception": false,
     "start_time": "2025-08-10T15:30:22.748017",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration:  0\n",
      "[6 9 3 ... 1 4 3] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.10329268292682926\n",
      "Iteration:  50\n",
      "[0 9 3 ... 0 2 3] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.3793170731707317\n",
      "Iteration:  100\n",
      "[9 9 3 ... 0 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.5718780487804878\n",
      "Iteration:  150\n",
      "[0 9 8 ... 0 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.6958292682926829\n",
      "Iteration:  200\n",
      "[9 9 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.7525121951219512\n",
      "Iteration:  250\n",
      "[5 5 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.7869756097560976\n",
      "Iteration:  300\n",
      "[5 5 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.8102682926829269\n",
      "Iteration:  350\n",
      "[5 5 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.8259024390243902\n",
      "Iteration:  400\n",
      "[5 5 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.8376097560975609\n",
      "Iteration:  450\n",
      "[5 5 8 ... 2 2 2] [5 5 8 ... 2 2 2]\n",
      "Accuracy:  0.8462195121951219\n"
     ]
    }
   ],
   "source": [
    "w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10ed71dd",
   "metadata": {
    "papermill": {
     "duration": 0.002655,
     "end_time": "2025-08-10T15:31:08.114812",
     "exception": false,
     "start_time": "2025-08-10T15:31:08.112157",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 861823,
     "sourceId": 3004,
     "sourceType": "competition"
    }
   ],
   "dockerImageVersionId": 31089,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 57.703209,
   "end_time": "2025-08-10T15:31:08.739245",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-08-10T15:30:11.036036",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
