#!/usr/bin/env python3
"""CPU checks for RoCEnante on a switchless ring (b12x.comm.roce_ring + scripts/ring_mesh/plan.py).

- B12X_ROCE_PEER_HCA_MAPS: one ';'-separated map per rank is picked by rank, B12X_ROCE_PEER_HCA_MAP
  still wins, a wrong count refuses.
- plan.py orients the ring from the fabric subnets (the head's physical port f0 is clockwise),
  accepts flipped ports within a PCIe domain, and translates sparkring's per-rank
  paths into peer maps for the TP rank order. The fixtures are a real four-Spark ring whose TP
  order runs the other way round the cables (TP 0,1,2,3 = ring 0,3,2,1).
- every map pairs path i at both ends on the same PCIe domain.
- in the image: the RoCEnante overlay selects the package from DSV41_ROCE_RING.

No GPU, torch or network. usage: python3 tests/test_roce_ring.py
"""
import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, *candidates):
    for c in candidates:
        if c.is_file():
            spec = importlib.util.spec_from_file_location(name, c)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(candidates)


# the package is fetched at build (scripts/fetch_runtime.sh): in the image, or after
# `scripts/fetch_runtime.sh runtime/fetched b12x` on a host
paths = _load("roce_ring_paths", Path("/opt/b12x/b12x/comm/roce_ring/_path_config.py"),
              ROOT / "runtime/fetched/b12x/b12x/comm/roce_ring/_path_config.py")
plan = _load("ring_mesh_plan", ROOT / "scripts/ring_mesh/plan.py")

HOSTS = ["s1", "s2", "s3", "s4"]  # TP rank order
# netdev -> IPv4 per host; each DAC is its own /24, f0 of every node cabled to the f1 of the node before it in TP order
ADDR = {
    "s1": {"enp1s0f0np0": "10.10.3.11/24", "enp1s0f1np1": "10.10.0.10/24", "enP2p1s0f0np0": "10.20.3.11/24", "enP2p1s0f1np1": "10.20.0.10/24"},
    "s2": {"enp1s0f0np0": "10.10.0.11/24", "enp1s0f1np1": "10.10.1.10/24", "enP2p1s0f0np0": "10.20.0.11/24", "enP2p1s0f1np1": "10.20.1.10/24"},
    "s3": {"enp1s0f0np0": "10.10.1.11/24", "enp1s0f1np1": "10.10.2.10/24", "enP2p1s0f0np0": "10.20.1.11/24", "enP2p1s0f1np1": "10.20.2.10/24"},
    "s4": {"enp1s0f0np0": "10.10.2.11/24", "enp1s0f1np1": "10.10.3.10/24", "enP2p1s0f0np0": "10.20.2.11/24", "enP2p1s0f1np1": "10.20.3.10/24"},
}
# sparkring rocenante_native_path_arguments() for that ring (ring rank -> "dest,function,rdma,gid,hops")
NATIVE = {
    "0": ["1,0,rocep1s0f0,3,1", "1,1,roceP2p1s0f0,3,1", "2,0,rocep1s0f0,3,2", "2,1,roceP2p1s0f1,3,2", "3,0,rocep1s0f1,3,1", "3,1,roceP2p1s0f1,3,1"],
    "1": ["0,0,rocep1s0f1,3,1", "0,1,roceP2p1s0f1,3,1", "2,0,rocep1s0f0,3,1", "2,1,roceP2p1s0f0,3,1", "3,0,rocep1s0f0,3,2", "3,1,roceP2p1s0f1,3,2"],
    "2": ["0,0,rocep1s0f1,3,2", "0,1,roceP2p1s0f0,3,2", "1,0,rocep1s0f1,3,1", "1,1,roceP2p1s0f1,3,1", "3,0,rocep1s0f0,3,1", "3,1,roceP2p1s0f0,3,1"],
    "3": ["0,0,rocep1s0f0,3,1", "0,1,roceP2p1s0f0,3,1", "1,0,rocep1s0f1,3,2", "1,1,roceP2p1s0f0,3,2", "2,0,rocep1s0f1,3,1", "2,1,roceP2p1s0f1,3,1"],
}
MAPS = "1=1/3,2=0/3,3=0/2;0=0/2,2=1/3,3=1/2;0=1/2,1=0/2,3=1/3;0=1/3,1=0/3,2=0/2"


def inventory(addr):
    return {h: {"ports": {nd: {"ipv4": ip} for nd, ip in ports.items()}} for h, ports in addr.items()}


class PeerMapEnv(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in (paths.PEER_HCA_MAP_ENV, paths.PEER_HCA_MAPS_ENV)}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_per_rank_selection(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        self.assertEqual(paths.peer_hca_map(4, 0, 4), ((-1, -1), (1, 3), (0, 3), (0, 2)))
        self.assertEqual(paths.peer_hca_map(4, 3, 4), ((1, 3), (0, 3), (0, 2), (-1, -1)))

    def test_single_map_wins(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        os.environ[paths.PEER_HCA_MAP_ENV] = "1=0/2,2=0/3,3=1/3"
        self.assertEqual(paths.peer_hca_map(4, 0, 4)[1], (0, 2))

    def test_wrong_count_refuses(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS.rsplit(";", 1)[0]
        with self.assertRaises(ValueError):
            paths.peer_hca_map(4, 0, 4)

    def test_two_hca_default_unchanged(self):
        self.assertEqual(paths.peer_hca_map(4, 1, 2), ((0, 1), (-1, -1), (0, 1), (0, 1)))


class Plan(unittest.TestCase):
    def test_ring_order(self):
        self.assertEqual(plan.ring_order(inventory(ADDR), HOSTS), ["s1", "s4", "s3", "s2"])

    def test_flipped_node_preserves_cycle(self):
        bad = {h: dict(p) for h, p in ADDR.items()}
        for prefix in ("enp1", "enP2p1"):
            a, b = prefix + "s0f0np0", prefix + "s0f1np1"
            bad["s2"][a], bad["s2"][b] = bad["s2"][b], bad["s2"][a]
        self.assertEqual(plan.ring_order(inventory(bad), HOSTS), ["s1", "s4", "s3", "s2"])

    def test_cross_domain_cable_refuses(self):
        bad = {h: dict(p) for h, p in ADDR.items()}
        a, b = "enp1s0f0np0", "enP2p1s0f0np0"
        bad["s2"][a], bad["s2"][b] = bad["s2"][b], bad["s2"][a]
        with self.assertRaises(SystemExit):
            plan.ring_order(inventory(bad), HOSTS)

    def test_flipped_topology_uses_actual_devices(self):
        addr = {h: dict(p) for h, p in ADDR.items()}
        for prefix in ("enp1", "enP2p1"):
            a, b = prefix + "s0f0np0", prefix + "s0f1np1"
            addr["s2"][a], addr["s2"][b] = addr["s2"][b], addr["s2"][a]
        inv = inventory(addr)
        for h, value in inv.items():
            for nd, p in value["ports"].items():
                p.update(rdma=nd.replace("en", "roce").replace("np0", "").replace("np1", ""), mac="00:00:00:00:00:01")
        order = plan.ring_order(inv, HOSTS)
        t = plan.topology(inv, order)
        rank = next(r for r in t["ranks"] if r["ssh_alias"] == "s2")
        self.assertEqual(rank["ports"]["clockwise"][0]["netdev"], "enp1s0f1np1")
        self.assertEqual(rank["ports"]["clockwise"][1]["rdma_device"], "roceP2p1s0f1")

    def test_peer_maps(self):
        order = plan.ring_order(inventory(ADDR), HOSTS)
        self.assertEqual(plan.peer_maps(NATIVE, order, HOSTS), MAPS)

    def test_maps_pair_paths_consistently(self):
        m = [dict((int(p), tuple(map(int, v.split("/")))) for p, v in (e.split("=") for e in r.split(","))) for r in MAPS.split(";")]
        domain, port = (lambda h: h // 2), (lambda h: h % 2)
        for r in range(4):
            for p in m[r]:
                for i in (0, 1):
                    self.assertEqual(domain(m[r][p][i]), domain(m[p][r][i]), (r, p, i))
                    if (p - r) % 4 in (1, 3):
                        self.assertNotEqual(port(m[r][p][i]), port(m[p][r][i]), (r, p, i))


class ImageOverlay(unittest.TestCase):
    PYNCCL = Path("/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py")

    @unittest.skipUnless(PYNCCL.is_file(), "image only")
    def test_overlay_selects_package(self):
        src = self.PYNCCL.read_text()
        self.assertIn('"b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce"', src)
        self.assertNotIn("from b12x.comm import roce\n", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
