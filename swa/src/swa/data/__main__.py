"""
数据项目管理 CLI — 创建、备份、查询项目。

用法:
    pixi run python -m swa.data.manager --list
    pixi run python -m swa.data.manager --create week1 data/raw/downloaded.jsonl
    pixi run python -m swa.data.manager --create week1 data/raw/downloaded.jsonl --desc "7月6日全量"
    pixi run python -m swa.data.manager --summary week1
    pixi run python -m swa.data.manager --backup week1 week1_bak
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import click
from swa.data import DataManager


@click.command()
@click.option("--list", "list_proj", is_flag=True, help="列出所有项目")
@click.option("--create", "create_name", default=None, help="创建项目名称")
@click.option("--source", default=None, help="来源 JSONL 文件路径")
@click.option("--desc", default="", help="项目描述")
@click.option("--summary", "sum_name", default=None, help="查看项目摘要")
@click.option("--backup", nargs=2, default=None, help="备份项目 原名称 备份名称")
def main(list_proj, create_name, source, desc, sum_name, backup):
    dm = DataManager()

    if list_proj:
        projects = dm.list_projects()
        if not projects:
            click.echo("暂无项目")
        else:
            click.echo(f"{'名称':<20s} {'记录数':<8s} {'启用':<8s} {'来源'}")
            click.echo("-" * 60)
            for p in projects:
                click.echo(f"{p.get('name',''):<20s} {p.get('total_records',0):<8d} {p.get('enabled_records',0):<8d} {p.get('source','')}")

    elif create_name:
        if not source:
            click.echo("--source 是必填的")
            return
        click.echo(f"创建项目 '{create_name}' 从 {source}...")
        meta = dm.create_project(create_name, source, desc)
        click.echo(f"完成: {meta['total_records']} 条")

    elif sum_name:
        dm.load_project(sum_name)
        s = dm.summary()
        click.echo(f"项目: {sum_name}")
        for k, v in s.items():
            click.echo(f"  {k}: {v}")

    elif backup:
        src, dst = backup
        click.echo(f"备份 {src} → {dst}...")
        dm.load_project(src)
        meta = dm.backup(dst)
        click.echo(f"完成")


if __name__ == "__main__":
    main()
