const transactionIdCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
});

function getTransactionId(item) {
  return String(item?.transaction_id || item?._id || item?.mongo_id || '');
}

function getCreatedAt(item) {
  return item?.created_at ? new Date(item.created_at).getTime() : 0;
}

export function sortTransactionsByTransactionIdDesc(items) {
  return [...items].sort((a, b) => {
    const aId = getTransactionId(a);
    const bId = getTransactionId(b);

    if (aId && bId) {
      const idOrder = transactionIdCollator.compare(bId, aId);
      if (idOrder !== 0) {
        return idOrder;
      }
    } else if (aId || bId) {
      return aId ? -1 : 1;
    }

    const createdAtOrder = getCreatedAt(b) - getCreatedAt(a);
    if (createdAtOrder !== 0) {
      return createdAtOrder;
    }

    return transactionIdCollator.compare(String(b?._id || b?.mongo_id || ''), String(a?._id || a?.mongo_id || ''));
  });
}
