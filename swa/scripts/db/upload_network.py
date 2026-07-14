"""
将 snmp.db 中的 YSH_OID 三张表上传到达梦数据库 YS_DB_HD 模式。

用法:
    pixi run python scripts/db/upload_network.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import click
import sqlite3
import dmPython


SQL = {
    "dev": "INSERT INTO YS_DB_HD.YSH_OID_DEVICE (DEVICE_ID, DEVICE_NAME, DEVICE_IP, DEVICE_TYPE) VALUES (?,?,?,?)",
    "port": "INSERT INTO YS_DB_HD.YSH_OID_PORT (PORT_ID, DEVICE_ID, PORT_NAME, PORT_INDEX, PORT_MAC, PORT_SPEED, PORT_STATUS, PORT_ADMIN, PORT_TYPE, PORT_MTU, PORT_DESCR, IS_AGGREGATION, MAC_COUNT) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
    "conn": "INSERT INTO YS_DB_HD.YSH_OID_CONN (CONN_ID, SOURCE_DEVICE_ID, SOURCE_DEVICE_NAME, SOURCE_DEVICE_TYPE, SOURCE_DEVICE_IP, SOURCE_PORT_ID, SOURCE_PORT_NAME, SOURCE_PORT_MAC, DEST_DEVICE_ID, DEST_DEVICE_NAME, DEST_DEVICE_TYPE, DEST_DEVICE_IP, DEST_PORT_ID, DEST_PORT_NAME, DEST_PORT_MAC) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
}


@click.command()
@click.option("--src", default="d:/project/oid/snmp.db")
@click.option("--host", default="10.21.1.190")
@click.option("--port", type=int, default=5236)
@click.option("--user", default="SYSDBA")
@click.option("--password", default="Yshed$888")
def main(src, host, port, user, password):
    # 1. 读取
    click.echo(f"从 {src} 读取 ...")
    con = sqlite3.connect(src)
    con.row_factory = sqlite3.Row
    c = con.cursor()

    c.execute("SELECT * FROM YSH_OID_DEVICE ORDER BY device_id")
    devices = [dict(r) for r in c.fetchall()]
    c.execute("SELECT * FROM YSH_OID_PORT ORDER BY port_id")
    ports = [dict(r) for r in c.fetchall()]
    c.execute("SELECT * FROM YSH_OID_CONN ORDER BY conn_id")
    conns = [dict(r) for r in c.fetchall()]
    con.close()

    click.echo(f"  DEVICE: {len(devices)} 行")
    click.echo(f"  PORT:   {len(ports)} 行")
    click.echo(f"  CONN:   {len(conns)} 行")

    if not password or password == "Yshed":
        password = "Yshed$888"

    # 2. 清空旧数据
    click.echo(f"\n连接到达梦 {host}:{port} ...")
    dm = dmPython.connect(user=user, password=password, server=host, port=port, autoCommit=True)
    cur = dm.cursor()
    click.echo("清空旧数据 ...")
    cur.execute("DELETE FROM YS_DB_HD.YSH_OID_CONN")
    cur.execute("DELETE FROM YS_DB_HD.YSH_OID_PORT")
    cur.execute("DELETE FROM YS_DB_HD.YSH_OID_DEVICE")

    click.echo("写入 YSH_OID_DEVICE ...")
    for d in devices:
        cur.execute(SQL["dev"], (d["device_id"], d["device_name"], d["device_ip"], d["device_type"]))

    click.echo("写入 YSH_OID_PORT ...")
    for p in ports:
        cur.execute(SQL["port"], (
            p["port_id"], p["device_id"], p["port_name"], p["port_index"],
            p["port_mac"], p["port_speed"], p["port_status"], p["port_admin"],
            p["port_type"], p["port_mtu"], p["port_descr"],
            p["is_aggregation"], p["mac_count"],
        ))

    click.echo("写入 YSH_OID_CONN ...")
    for cn in conns:
        cur.execute(SQL["conn"], (
            cn["conn_id"],
            cn["source_device_id"], cn["source_device_name"], cn["source_device_type"], cn["source_device_ip"],
            cn["source_port_id"], cn["source_port_name"], cn["source_port_mac"],
            cn["dest_device_id"], cn["dest_device_name"], cn["dest_device_type"], cn["dest_device_ip"],
            cn["dest_port_id"], cn["dest_port_name"], cn["dest_port_mac"],
        ))

    dm.close()
    click.echo("完成!")


if __name__ == "__main__":
    main()
