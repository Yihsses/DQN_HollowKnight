using System;
using System.Collections;
using System.Collections.Generic;
using GlobalEnums;
using Modding;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using static UnityEngine.UI.GridLayoutGroup;

namespace MyFirstMod
{
    public class MyFirstMod : Mod
    {
        private TcpListener listener;
        private TcpClient client;
        private NetworkStream stream;

        public MyFirstMod() : base("My First Mod") { }

        public override string GetVersion() => "v1";

        public override void Initialize()
        {
            // 啟動 Socket Server
            listener = new TcpListener(IPAddress.Any, 5555);
            listener.Start();

            // 非同步接受連線，避免阻塞主執行緒
            listener.BeginAcceptTcpClient(new AsyncCallback(AcceptCallback), null);
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            ModHooks.TakeDamageHook += GetHpDamage;
        }

        private void AcceptCallback(IAsyncResult ar)
        {
            client = listener.EndAcceptTcpClient(ar);
            stream = client.GetStream();
        }

        public void OnHeroUpdate()
        {
            if (stream == null || !stream.CanWrite) return;

            var boss = BossSceneController.Instance;
            var hero = HeroController.instance;
            //if (fsm.gameObject.name.Contains("Hornet"))
            //{
            //    // 攔截特定技能進入時
            //    Log(fsm.ActiveStateName);
            //}

            StringBuilder sb = new StringBuilder();

            if (boss != null && boss.bosses != null)
            {

                foreach (var b in boss.bosses)
                {
                    if (b != null)
                    {
                        int currentHP = b.hp;
                        Vector2 currentPosition = b.transform.position;
                        sb.Append($"{currentHP}/{currentPosition}/{hero.playerData.GetInt("health")}/{hero.transform.position}/{hero.playerData.GetInt("MPCharge")}");
                    }
                    // 找出所有附加在這個 boss 上的 FSM
                }
            }

            //var hc = HeroController.instance;
            //if (hc != null)
            //{
            //    Vector2 currentPosition = hc.transform.position;
            //    sb.Append($"Hero Pos: {currentPosition}");
            //}

            // 將字串轉為 bytes 傳出
            byte[] data = Encoding.UTF8.GetBytes(sb.ToString() + "\n");

            try
            {
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch
            {
                // 連線斷掉，重設 stream，等待新的連線
                stream = null;
                client = null;
                listener.BeginAcceptTcpClient(new AsyncCallback(AcceptCallback), null);
            }
        }

        public int GetHpDamage(ref int hazardType, int damage)
        {
            //Log($"HazardType: {hazardType}, Damage: {damage}");
            return damage;
        }

        public  void CloseSocket()
        {
            listener?.Stop();
            client?.Close();
        }
    }
}