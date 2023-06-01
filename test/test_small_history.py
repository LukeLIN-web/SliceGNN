from timeit import default_timer as timer

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.testing.decorators import withCUDA

from microGNN import History
from microGNN.models import SAGE, ScaleSAGE
from microGNN.prune import prune_computation_graph
from microGNN.utils import get_nano_batch_histories
from microGNN.utils.common_class import Adj

hop = [-1, -1]
num_layers = 2
in_channels = 8
hidden_channels = 4
out_channels = 2
node_num = 8
features = [[i for j in range(in_channels)] for i in range(node_num)]
labels = [i % 2 for i in range(node_num)]
# yapf: disable
edge_index = torch.tensor([[2, 3, 3, 4, 5, 6, 7],
                            [0, 0, 1, 1, 2, 3, 4]], dtype=torch.long) # noqa
# yapf: enable
torch.manual_seed(23)
mb_n_id = torch.arange(node_num)


@withCUDA
def test_same_out(device):
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    batch_size, n_id, adjs = next(iter(train_loader))
    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, len(n_id), hidden_channels, device)
        for cacheid in cached_id
    ])
    nb = nano_batchs[0]
    adjs = [adj.to(device) for adj in nb.adjs]
    x = torch.tensor(features, dtype=torch.float).to(device)

    model1 = ScaleSAGE(in_channels, hidden_channels, out_channels,
                       num_layers).to(device)
    nb1id = nb.n_id.to(device)
    out1 = model1(x[n_id][nb1id], nb1id, adjs, histories)

    model2 = SAGE(in_channels, hidden_channels, out_channels,
                  num_layers).to(device)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(x[n_id][nb.n_id], adjs)
    assert torch.equal(out1, out2)

    target_node = n_id[:batch_size]
    y = torch.tensor(labels, dtype=torch.long).to(device)
    loss1 = F.nll_loss(out1, y[target_node][:nb.size])
    loss2 = F.nll_loss(out2, y[target_node][:nb.size])
    loss1.backward()
    loss2.backward()

    grad1 = [param.grad.clone().view(-1) for param in model1.parameters()]
    grad2 = [param.grad.clone().view(-1) for param in model2.parameters()]
    assert torch.equal(torch.cat(grad1), torch.cat(grad2))

    nb1 = nano_batchs[1]
    adjs = [adj.to(device) for adj in nb1.adjs]
    nb1id = nb1.n_id.to(device)
    out1 = model1(x[n_id][nb1id], nb1id, adjs, histories)
    out2 = model2(x[n_id][nb1.n_id], adjs)
    assert torch.equal(out1, out2)
    loss1 = F.nll_loss(out1, y[target_node][nb1.size:2 * nb1.size])
    loss2 = F.nll_loss(out2, y[target_node][nb1.size:2 * nb1.size])
    assert torch.equal(loss1, loss2)
    loss1.backward()
    loss2.backward()
    grad1 = [param.grad.clone().view(-1) for param in model1.parameters()]
    grad2 = [param.grad.clone().view(-1) for param in model2.parameters()]
    # only first hop gradient keeped
    assert torch.equal(torch.cat(grad1[-3:]), torch.cat(grad2[-3:]))


@withCUDA
def test_gradient(device):
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    batch_size, n_id, adjs = next(iter(train_loader))
    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, device)
        for cacheid in cached_id
    ]).to(device)
    nb = nano_batchs[0]
    adjs = [adj.to(device) for adj in nb.adjs]
    x = torch.tensor(features, dtype=torch.float).to(device)

    model1 = ScaleSAGE(in_channels, hidden_channels, out_channels,
                       num_layers).to(device)
    nbid = nb.n_id.to(device)
    out1 = model1(x[n_id][nbid], nbid, adjs, histories)
    target_node = n_id[:batch_size]
    y = torch.tensor(labels, dtype=torch.long).to(device)
    loss1 = F.nll_loss(out1, y[target_node][:nb.size])
    loss1.backward()
    for param in model1.parameters():
        assert param.grad is not None


def test_small_save_embedding():
    train_loader = NeighborSampler(
        edge_index,
        sizes=hop,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    batch_size, n_id, adjs = next(iter(train_loader))
    model = ScaleSAGE(in_channels, hidden_channels, out_channels, num_layers)
    model.eval()
    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)

    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    assert torch.equal(histories[0].emb_idx,
                       torch.tensor([-1, -1, -1, 0, -1, -1, -1, -1]))

    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))
    nb = nano_batchs[0]
    x = torch.tensor(features, dtype=torch.float)
    model(x[n_id][nb.n_id], nb.n_id, nb.adjs, histories)
    assert torch.equal(
        histories[0].cached_nodes,
        torch.tensor([False, False, False, True, False, False, False, False]))
    assert not torch.equal(histories[0].emb[0], torch.zeros(4))


def test_small_histfunction():
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    nb = nano_batchs[1]

    x = torch.tensor(features, dtype=torch.float)
    x = x[mb_n_id][nb.n_id]
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    histories[0].emb[0] = torch.tensor([3.3, 3.4, 3.5, 3.6])  # should be pull
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)
    for i, adj in enumerate(pruned_adjs):
        batch_size = nb.adjs[i].size[1]
        x_target = x[:batch_size]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            assert torch.equal(x[1], torch.zeros(4))
            x = history.pull(x, nb.n_id[:batch_size])
            assert not torch.equal(x[0], torch.tensor([3.3, 3.4, 3.5, 3.6]))
            assert torch.equal(x[1], torch.tensor([3.3, 3.4, 3.5, 3.6]))
            history.push(x, nb.n_id[:batch_size])
    loss = F.nll_loss(x, torch.tensor([1]))
    loss.backward()
    for param in convs.parameters():
        assert param.grad is not None


def test_small_pull():
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    histories[0].cached_nodes = torch.tensor(
        [False, False, False, True, False, False, False, False])
    histories[0].emb[0] = torch.tensor([3.3, 3.4, 3.5, 3.6])  # should be pull

    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)

    x = torch.tensor(features, dtype=torch.float)
    x = x[mb_n_id][nb.n_id]

    for i, adj in enumerate(pruned_adjs):
        batch_size = nb.adjs[i].size[1]
        x_target = x[:batch_size]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            out = x.clone()
            assert torch.equal(out[1], torch.zeros(4))
            for j, id in enumerate(nb.n_id[:batch_size]):
                if history.cached_nodes[id]:
                    embidx = history.emb_idx[id]
                    assert embidx == 0
                    emb = history.emb[embidx]
                    out[j] = emb
            assert not torch.equal(out[0], torch.tensor([3.3, 3.4, 3.5, 3.6]))
            assert torch.equal(out[1], torch.tensor([3.3, 3.4, 3.5, 3.6]))
    loss = F.nll_loss(x, torch.tensor([1]))
    loss.backward()
    for param in convs.parameters():
        assert param.grad is not None


def test_small_push():
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    nb = nano_batchs[1]
    pruned_adjs = prune_computation_graph(nb.n_id, nb.adjs, histories)
    x = torch.tensor(features, dtype=torch.float)
    x = x[mb_n_id][nb.n_id]

    for i, adj in enumerate(pruned_adjs):
        batch_size = adj.size[1]
        x_target = x[:batch_size]  # require 前size[0]个节点是 layer nodes
        x = convs[i]((x, x_target),
                     adj.edge_index)  # compute the non cached nodes embedding
        if i != num_layers - 1:  # last layer is not saved
            history = histories[i]
            assert torch.equal(histories[0].emb[0], torch.zeros(4))
            for j, id in enumerate(nb.n_id[:batch_size]):
                if history.emb_idx[id] != -1:
                    embidx = history.emb_idx[id]
                    assert embidx == 0
                    history.emb[embidx] = x[j]
                    history.cached_nodes[id] = True
            assert not torch.equal(histories[0].emb[0], torch.zeros(4))
            assert torch.equal(
                history.cached_nodes,
                torch.tensor(
                    [False, False, False, True, False, False, False, False]))
    loss = F.nll_loss(x, torch.tensor([1]))
    loss.backward()
    for param in convs.parameters():
        assert param.grad is not None


def test_init_history():
    edge1 = torch.tensor([[2, 3, 3, 4], [0, 0, 1, 1]])
    adjs1 = Adj(edge1, None, (5, 2))
    edge2 = torch.tensor([[2, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 3, 4]])
    adjs2 = Adj(edge2, None, (8, 5))
    adjs = [adjs2, adjs1]
    convs = torch.nn.ModuleList()
    convs.append(
        SAGEConv(in_channels, hidden_channels, root_weight=False, bias=False))
    convs.append(
        SAGEConv(hidden_channels, out_channels, root_weight=False, bias=False))

    nano_batchs, cached_id = get_nano_batch_histories(adjs,
                                                      mb_n_id,
                                                      batch_size=2,
                                                      num_nano_batch=2)
    histories = torch.nn.ModuleList([
        History(cacheid, node_num, hidden_channels, 'cpu')
        for cacheid in cached_id
    ])
    assert torch.equal(
        histories[0].cached_nodes,
        torch.tensor([False, False, False, False, False, False, False, False]))
    assert histories[0].emb.size() == (1, 4)
    assert torch.equal(histories[0].emb_idx,
                       torch.tensor([-1, -1, -1, 0, -1, -1, -1, -1]))


if __name__ == "__main__":
    # test_small_save_embedding()
    # test_small_histfunction()
    # test_small_push()
    # test_small_pull()
    test_same_out('cpu')
