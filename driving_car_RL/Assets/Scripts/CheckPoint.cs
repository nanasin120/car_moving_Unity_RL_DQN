using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckPoint : MonoBehaviour
{
    public static CheckPoint instance;
    public GameObject[] checkpoint;
    public int now = 0;

    void Start()
    {
        instance = this;
    }

    public void next()
    {
        checkpoint[now].SetActive(false);
        now = (now + 1) % checkpoint.Length;
        checkpoint[now].SetActive(true);
    }

    public void restart()
    {
        for(int i = 0; i < checkpoint.Length; i++) checkpoint[i].SetActive(false);
        now = 0;
        checkpoint[now].SetActive(true);
    }
}
