import torch


def eval_task(model, args, test_data, test_labels, task, test_bs=1000):
    """
    Evaluate model on a set of tasks and return accuracy and average confidence.
    """
    model.eval()
    total = 0
    correct = 0
    confidence_sum = 0.0
    for i in range(0, len(test_data), test_bs):
        current_data = torch.Tensor(test_data.reshape(-1, 32*32*3)).float()
        current_labels = torch.Tensor(test_labels).long()
        if args.cuda:
            current_data, current_labels = current_data.cuda(), current_labels.cuda()
        output = model.forward(current_data, task)
        pred = output.data.max(1)[1]
        correct += (pred == current_labels).sum().item()
        total += current_labels.size(0)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence_sum += probabilities.max(1)[0].sum().item()
    accuracy = correct / total
    avg_confidence = confidence_sum / total
    return accuracy, avg_confidence

def eval_retain_forget_test(model, args, retain_set, forget_set, test_set, retain_acc, forget_acc, test_acc, test_acc_forget):
    test_bs = 1000
    correct_retain = 0
    total_retain = 0
    for i in range(0, len(retain_set)):
        correct = 0
        total = len(retain_set[i])     

        # Test the model
        
        x = torch.Tensor(retain_set[i][0].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(retain_set[i][1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        correct = 0
        total = len(retain_set[i][0])
        for j in range(0,len(retain_set[i][0]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_retain += correct
        total_retain += total
    print("Total correct retain: ", correct_retain, " Total retain: ", total_retain)
    
    retain_acc.append(correct_retain / total_retain)
    
    correct_forget = 0
    total_forget = 0
    for i in range(0, len(forget_set)):
        correct = 0
        total = len(forget_set[i])     

        # Test the model
        
        x = torch.Tensor(forget_set[i][0].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(forget_set[i][1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        total = len(forget_set[i][0])
        for j in range(0,len(forget_set[i][0]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i + len(retain_set))
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_forget += correct
        total_forget += total
    print("Total correct forget: ", correct_forget, " Total forget: ", total_forget)
    
    forget_acc.append(correct_forget / total_forget)
    
    correct_test = 0
    total_test = 0
    
    for i in range(0, 2 * len(retain_set), 2):
        correct = 0
        total = len(test_set[i])     

        # Test the model
        
        x = torch.Tensor(test_set[i].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(test_set[i+1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        average_confidence_task = []
        # keep track of average confidence score
        for j in range(0,len(test_set[i]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i // 2)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_test += correct
        total_test += total
    print("Total correct test: ", correct_test, " Total test: ", total_test)
        
    test_acc.append(correct_test/total_test)
    
    correct_test_forget = 0
    total_test_forget = 0
    for i in range(2 * len(retain_set), 2 * (len(retain_set) + len(forget_set)),  2):
        correct = 0
        total = len(test_set[i])     

        # Test the model
        
        x = torch.Tensor(test_set[i].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(test_set[i+1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        average_confidence_task = []
        # keep track of average confidence score
        for j in range(0,len(test_set[i]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i // 2)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_test_forget += correct
        total_test_forget += total
        
    test_acc_forget.append(correct_test_forget/total_test_forget)
    
    print("Total correct forget test: ", correct_test_forget, " Total test forget: ", total_test_forget)
        
    return retain_acc, forget_acc, test_acc, test_acc_forget