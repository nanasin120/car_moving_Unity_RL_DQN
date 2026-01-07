using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class testAgent : Agent
{
    public float moveSpeed = 0.1f, maxMoveSpeed = 5f;
    public float turnSpeed = 10f;
    public Rigidbody rb;
    public override void Initialize()
    {
        MaxStep = 5000;

        //rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
        transform.GetComponent<Rigidbody>().velocity = transform.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        CheckPoint.instance.restart();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(rb.velocity.magnitude);
        sensor.AddObservation(rb.velocity);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];
        //Debug.Log($"[0] = {action[0]}, [1]: {action[1]}");

        Vector3 dir = Vector3.zero;
        float rot = 0f;

        switch (action)
        {
            case 1: dir = Vector3.forward; break;
            case 2: dir = -Vector3.forward; break;
            case 3: rot = -1f; break;
            case 4: rot = 1; break;
        }

        Vector3 angularVelocity = new Vector3(0, rot * turnSpeed, 0);
        rb.angularVelocity = angularVelocity;

        rb.AddRelativeForce(dir * moveSpeed, ForceMode.Acceleration);

        if(rb.velocity.magnitude > maxMoveSpeed)
        {
            rb.velocity = rb.velocity.normalized * maxMoveSpeed;
        }

        AddReward((rb.velocity.magnitude - 2.5f) / 5f);
    }

    public override void Heuristic(in ActionBuffers actionOut)
    {
        var action = actionOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W)) action[0] = 1;
        else if (Input.GetKey(KeyCode.S)) action[0] = 2;
        else if (Input.GetKey(KeyCode.A)) action[0] = 3;
        else if (Input.GetKey(KeyCode.D)) action[0] = 4;
    }

    void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.tag == "Wall")
        {
            AddReward(-100f);
            EndEpisode();
        }
    }

    void OnTriggerEnter(Collider collision)
    {
        if(collision.gameObject.tag == "CheckPoint")
        {
            AddReward(50f);
            CheckPoint.instance.next();
        }
    }
}