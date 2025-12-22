---
sidebar_position: 4
title: "Unity Integration"
---

# Unity for High-Fidelity Robot Simulation

## Overview

While Gazebo excels at physics simulation, **Unity** provides photorealistic rendering essential for training vision-based AI systems. Unity's High Definition Render Pipeline (HDRP) creates visuals indistinguishable from reality, reducing the visual domain gap in sim-to-real transfer.

## Why Unity for Robotics?

| Capability | Benefit for Humanoids |
|------------|----------------------|
| **Photorealistic Rendering** | Train vision models that transfer to reality |
| **Human Avatars** | Realistic human-robot interaction scenarios |
| **Dynamic Environments** | Procedurally generated training worlds |
| **VR/AR Integration** | Teleoperation and immersive debugging |
| **Asset Store** | Thousands of 3D models and environments |

## Unity-ROS 2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Unity-ROS 2 Integration                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   UNITY SIDE                        ROS 2 SIDE              │
│   ┌──────────────┐                  ┌──────────────┐        │
│   │  HDRP Scene  │                  │  ROS 2 Nodes │        │
│   │  + Robot     │                  │  (rclpy)     │        │
│   └──────┬───────┘                  └──────┬───────┘        │
│          │                                 │                 │
│   ┌──────▼───────┐                  ┌──────▼───────┐        │
│   │ ROS-TCP-     │◀───────────────▶│ ROS-TCP-     │        │
│   │ Connector    │    TCP/IP        │ Endpoint     │        │
│   └──────────────┘                  └──────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Setting Up Unity for Robotics

### Step 1: Install Unity Hub and Create Project

```bash
# Download Unity Hub from unity.com
# Create new project with HDRP template
# Unity 2022.3 LTS recommended
```

### Step 2: Install ROS-TCP-Connector

1. Open **Window → Package Manager**
2. Click **+ → Add package from git URL**
3. Enter: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector`

### Step 3: Configure ROS Settings

```
# In Unity: Robotics → ROS Settings
ROS IP Address: 127.0.0.1
ROS Port: 10000
Protocol: ROS2
```

### Step 4: Install ROS-TCP-Endpoint (ROS 2 side)

```bash
# Clone the endpoint package
cd ~/ros2_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git -b ROS2

# Build
cd ~/ros2_ws
colcon build --packages-select ros_tcp_endpoint

# Run endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1
```

## Importing Robot URDF into Unity

### Step 1: Install URDF Importer

Add package: `https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer`

### Step 2: Import Humanoid URDF

```csharp
// In Unity Editor:
// Assets → Import Robot from URDF
// Select your humanoid.urdf file
// Configure joint limits and collision meshes
```

### Unity URDF Import Settings

| Setting | Recommended Value |
|---------|-------------------|
| **Mesh Decomposer** | VHACD (convex decomposition) |
| **Convex Mesh** | Yes (for collision) |
| **Create Colliders** | Yes |
| **Joint Type** | Articulation Body |

## Publishing Sensor Data from Unity

### Camera Publisher

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

public class CameraPublisher : MonoBehaviour
{
    public Camera robotCamera;
    public string topicName = "/unity/camera/image_raw";
    public int publishRate = 30;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float publishInterval;
    private float timeSincePublish;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName);

        // Setup render texture
        renderTexture = new RenderTexture(640, 480, 24);
        texture2D = new Texture2D(640, 480, TextureFormat.RGB24, false);
        robotCamera.targetTexture = renderTexture;

        publishInterval = 1.0f / publishRate;
    }

    void Update()
    {
        timeSincePublish += Time.deltaTime;

        if (timeSincePublish >= publishInterval)
        {
            PublishImage();
            timeSincePublish = 0;
        }
    }

    void PublishImage()
    {
        // Capture from render texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;

        // Create ROS message
        byte[] imageData = texture2D.GetRawTextureData();

        ImageMsg msg = new ImageMsg
        {
            header = new HeaderMsg
            {
                stamp = new TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_link"
            },
            height = 480,
            width = 640,
            encoding = "rgb8",
            is_bigendian = 0,
            step = 640 * 3,
            data = imageData
        };

        ros.Publish(topicName, msg);
    }
}
```

### Depth Camera Publisher

```csharp
public class DepthPublisher : MonoBehaviour
{
    public Camera depthCamera;
    public string topicName = "/unity/depth/image_raw";

    private Shader depthShader;
    private Material depthMaterial;
    private RenderTexture depthTexture;

    void Start()
    {
        // Use depth shader for accurate depth extraction
        depthShader = Shader.Find("Hidden/DepthCapture");
        depthMaterial = new Material(depthShader);

        depthCamera.depthTextureMode = DepthTextureMode.Depth;
        depthTexture = new RenderTexture(640, 480, 24, RenderTextureFormat.RFloat);
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        Graphics.Blit(src, depthTexture, depthMaterial);

        // Convert to ROS PointCloud2 or depth image
        PublishDepth(depthTexture);

        Graphics.Blit(src, dest);
    }
}
```

## Receiving Commands in Unity

### Joint Command Subscriber

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class JointCommandSubscriber : MonoBehaviour
{
    public ArticulationBody[] joints;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<JointStateMsg>("/humanoid/joint_commands", ExecuteJointCommand);
    }

    void ExecuteJointCommand(JointStateMsg msg)
    {
        for (int i = 0; i < msg.position.Length && i < joints.Length; i++)
        {
            // Set target position for articulation body
            ArticulationDrive drive = joints[i].xDrive;
            drive.target = (float)(msg.position[i] * Mathf.Rad2Deg);
            joints[i].xDrive = drive;
        }
    }
}
```

## Human-Robot Interaction Scenarios

### Animating Human Avatars

```csharp
public class HumanAvatarController : MonoBehaviour
{
    public Animator humanAnimator;
    public Transform robotTransform;

    // Simulate natural human behavior
    void Update()
    {
        // Look at robot when nearby
        float distance = Vector3.Distance(transform.position, robotTransform.position);

        if (distance < 3.0f)
        {
            // Trigger look animation
            humanAnimator.SetBool("isLookingAtRobot", true);

            // Random gestures
            if (Random.value < 0.001f)
            {
                humanAnimator.SetTrigger("wave");
            }
        }
        else
        {
            humanAnimator.SetBool("isLookingAtRobot", false);
        }
    }
}
```

### Dynamic Environment Generation

```csharp
public class EnvironmentRandomizer : MonoBehaviour
{
    public GameObject[] furniturePrefabs;
    public Light[] sceneLights;
    public Material[] floorMaterials;

    // Call this between training episodes
    public void RandomizeEnvironment()
    {
        // Randomize lighting
        foreach (Light light in sceneLights)
        {
            light.intensity = Random.Range(0.5f, 2.0f);
            light.color = new Color(
                Random.Range(0.8f, 1.0f),
                Random.Range(0.8f, 1.0f),
                Random.Range(0.8f, 1.0f)
            );
        }

        // Randomize floor texture
        GetComponent<Renderer>().material =
            floorMaterials[Random.Range(0, floorMaterials.Length)];

        // Randomize furniture placement
        RandomizeFurniture();
    }
}
```

## Domain Randomization in Unity

Domain randomization is critical for sim-to-real transfer:

```csharp
public class DomainRandomizer : MonoBehaviour
{
    [Header("Visual Randomization")]
    public bool randomizeTextures = true;
    public bool randomizeLighting = true;
    public bool randomizeColors = true;

    [Header("Physics Randomization")]
    public bool randomizeFriction = true;
    public bool randomizeMass = true;

    [Header("Camera Randomization")]
    public bool randomizeExposure = true;
    public bool addNoise = true;
    public float noiseStrength = 0.02f;

    public void ApplyRandomization()
    {
        if (randomizeLighting)
        {
            // Vary lighting direction, intensity, color
            Light sun = FindObjectOfType<Light>();
            sun.transform.rotation = Quaternion.Euler(
                Random.Range(30, 70),
                Random.Range(0, 360),
                0
            );
        }

        if (addNoise)
        {
            // Apply camera noise post-processing
            ApplyCameraNoise();
        }

        if (randomizeFriction)
        {
            // Vary physics materials
            PhysicMaterial[] mats = FindObjectsOfType<PhysicMaterial>();
            foreach (var mat in mats)
            {
                mat.dynamicFriction = Random.Range(0.3f, 0.8f);
                mat.staticFriction = Random.Range(0.4f, 0.9f);
            }
        }
    }
}
```

## Performance Optimization

| Optimization | Implementation |
|--------------|----------------|
| **GPU Instancing** | Enable for repeated objects |
| **LOD Groups** | Reduce polygon count at distance |
| **Occlusion Culling** | Don't render hidden objects |
| **Fixed Timestep** | Match physics rate (0.01s = 100Hz) |
| **Async GPU Readback** | Non-blocking sensor data capture |

## Key Takeaways

1. **Unity + HDRP** provides photorealistic visuals for vision AI training
2. **ROS-TCP-Connector** enables seamless ROS 2 communication
3. **URDF Importer** converts robot models to Unity ArticulationBodies
4. **Domain randomization** in Unity improves sim-to-real transfer
5. **Human avatars** enable realistic interaction scenario training

---

*Next: Learn about simulating sensors in Gazebo and Unity.*
