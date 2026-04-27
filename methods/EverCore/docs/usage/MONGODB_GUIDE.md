# MongoDB Installation Guide

This document provides guidance for MemSys project developers on installing and configuring MongoDB directly on macOS, Windows, and Linux.

## 1. Install MongoDB Community Server

Please choose the installation method corresponding to your operating system.

### 1.1. For macOS Users

We recommend using [Homebrew](https://brew.sh/) for installation. Homebrew is a package manager for macOS that greatly simplifies the software installation process.

1.  **Install Homebrew** (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2.  **Add MongoDB's Homebrew Tap**:
    ```bash
    brew tap mongodb/brew
    ```

3.  **Install MongoDB**:
    ```bash
    brew install mongodb-community
    ```

### 1.2. For Windows Users

#### Method 1: Using the Official Installer (Recommended)
1.  Go to the [MongoDB Community Server Download Page](https://www.mongodb.com/try/download/community).
2.  Select the latest version `MSI` package and download it.
3.  Run the installer. During the installation process, **be sure to check the "Install MongoDB Compass"** option to install the graphical management tool at the same time.
4.  Follow the wizard to complete the installation. The installer will automatically set up MongoDB as a Windows service, which will start by default on boot.

#### Method 2: Using the Chocolatey Package Manager
If you use [Chocolatey](https://chocolatey.org/), you can install it by running the following command:
```powershell
choco install mongodb
```

### 1.3. For Linux Users (e.g., Ubuntu 20.04/22.04 LTS)

1.  **Import MongoDB's GPG Key**:
    ```bash
    sudo apt-get install gnupg
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
       sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
       --dearmor
    ```

2.  **Create a List File for MongoDB**:
    ```bash
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    ```

3.  **Update the Local Package Database and Install MongoDB**:
    ```bash
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    ```

## 2. Install MongoDB Compass (Graphical Tool)

If you did not select the bundled installation during the Windows setup, or if you are a macOS/Linux user, you can install Compass separately.

*   **macOS**:
    ```bash
    brew install --cask mongodb-compass
    ```

*   **Windows**:
    Go to the [MongoDB Compass Download Page](https://www.mongodb.com/try/download/compass), download the `MSI` package, and execute it.

*   **Linux (Ubuntu)**:
    Go to the [MongoDB Compass Download Page](https://www.mongodb.com/try/download/compass), download the `deb` format package, and then install it with the following command (please replace the filename with the actual name of the file you downloaded):
    ```bash
    sudo dpkg -i mongodb-compass_x.x.x_amd64.deb
    ```

## 3. Running the MongoDB Service

*   **macOS (via Homebrew)**:
    ```bash
    brew services start mongodb-community
    ```

*   **Windows**:
    If you installed via the official installer, MongoDB will be registered as a Windows service and run automatically. You can find and manage it in the "Services" application.

*   **Linux (Ubuntu)**:
    ```bash
    sudo systemctl start mongod
    # Enable auto-start on boot
    sudo systemctl enable mongod
    ```
    You can check the service status with `sudo systemctl status mongod`.

## 4. Connection and Configuration

### 4.1. Connecting with MongoDB Compass

Regardless of the operating system, open your installed MongoDB Compass. It will usually automatically detect your locally running MongoDB instance. You just need to click the "Connect" button and use the default connection string `mongodb://localhost:27017` to connect successfully.

### 4.2. Configuring the Project `.env` File

To allow the MemSys application to connect to the local MongoDB, you need to configure the project's environment variables.

1.  First, in the project's root directory, copy the `env.template` file to `.env` (if you haven't already):
    ```bash
    cp env.template .env
    ```

2.  Open the `.env` file and find the MongoDB configuration section. Since a local installation of MongoDB does not have user authentication enabled by default, you need to make the following changes:

    ```ini
    # ===================
    # MongoDB Configuration
    # ===================

    # Tenant ID (required)
    TENANT_SINGLE_TENANT_ID=t_yourname

    MONGODB_HOST=127.0.0.1
    MONGODB_PORT=27017
    MONGODB_USERNAME=
    MONGODB_PASSWORD=
    MONGODB_URI_PARAMS=
    ```
    **Note**: Leave `MONGODB_USERNAME` and `MONGODB_PASSWORD` blank, and also set `MONGODB_URI_PARAMS` to empty. The actual database name is auto-generated from `TENANT_SINGLE_TENANT_ID` (e.g., `t_yourname_memsys`).

Now, your development environment is configured, and the MemSys application can successfully connect to the locally running MongoDB instance.
