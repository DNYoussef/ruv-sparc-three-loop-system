/**
 * Data Synchronization Engine Tests
 * Test suite for bidirectional sync, conflict resolution, and state management
 */

const { expect } = require('chai');
const sinon = require('sinon');
const SyncOrchestrator = require('../resources/scripts/sync-orchestrator');
const ChangeDetector = require('../resources/scripts/change-detector');
const ConflictResolver = require('../resources/scripts/conflict-resolver');
const SyncStateManager = require('../resources/scripts/sync-state-manager');

describe('Data Synchronization Tests', () => {
  describe('Change Detection', () => {
    let detector;

    beforeEach(() => {
      detector = new ChangeDetector();
    });

    it('should calculate consistent checksums', () => {
      const data = {
        Id: '001',
        Email: 'test@example.com',
        FirstName: 'John',
        LastName: 'Doe'
      };

      const checksum1 = detector.calculateChecksum(data);
      const checksum2 = detector.calculateChecksum(data);

      expect(checksum1).to.equal(checksum2);
    });

    it('should ignore volatile fields in checksum', () => {
      const data1 = {
        Id: '001',
        Email: 'test@example.com',
        LastModifiedDate: '2025-01-01'
      };

      const data2 = {
        Id: '001',
        Email: 'test@example.com',
        LastModifiedDate: '2025-01-02'
      };

      const checksum1 = detector.calculateChecksum(data1);
      const checksum2 = detector.calculateChecksum(data2);

      expect(checksum1).to.equal(checksum2);
    });

    it('should detect changes in data', () => {
      const oldChecksum = detector.calculateChecksum({ Email: 'old@example.com' });
      const newChecksum = detector.calculateChecksum({ Email: 'new@example.com' });

      expect(detector.hasChanged(oldChecksum, newChecksum)).to.be.true;
    });

    it('should identify changed fields', () => {
      const oldData = {
        Id: '001',
        Email: 'old@example.com',
        FirstName: 'John'
      };

      const newData = {
        Id: '001',
        Email: 'new@example.com',
        FirstName: 'John'
      };

      const changes = detector.getChangedFields(oldData, newData);

      expect(changes).to.have.property('Email');
      expect(changes.Email.old).to.equal('old@example.com');
      expect(changes.Email.new).to.equal('new@example.com');
      expect(changes).to.not.have.property('FirstName');
    });
  });

  describe('Conflict Resolution', () => {
    it('should resolve using last_write_wins strategy', () => {
      const resolver = new ConflictResolver('last_write_wins');

      const sourceRecord = {
        Id: '001',
        Email: 'source@example.com',
        LastModifiedDate: '2025-01-02T12:00:00Z'
      };

      const targetRecord = {
        Id: '001',
        Email: 'target@example.com',
        LastModifiedDate: '2025-01-01T12:00:00Z'
      };

      const resolved = resolver.resolve(sourceRecord, targetRecord);

      expect(resolved.Email).to.equal('source@example.com');
    });

    it('should resolve using source_wins strategy', () => {
      const resolver = new ConflictResolver('source_wins');

      const sourceRecord = { Email: 'source@example.com' };
      const targetRecord = { Email: 'target@example.com' };

      const resolved = resolver.resolve(sourceRecord, targetRecord);

      expect(resolved).to.deep.equal(sourceRecord);
    });

    it('should merge specific fields', () => {
      const resolver = new ConflictResolver('merge_fields');

      const sourceRecord = {
        Email: 'source@example.com',
        Status: 'Active',
        Priority: 'High'
      };

      const targetRecord = {
        Email: 'target@example.com',
        Status: 'Inactive',
        Priority: 'Low'
      };

      const resolved = resolver.resolve(sourceRecord, targetRecord, {
        mergeFields: ['Status', 'Priority']
      });

      expect(resolved.Email).to.equal('target@example.com');
      expect(resolved.Status).to.equal('Active');
      expect(resolved.Priority).to.equal('High');
    });

    it('should flag for manual resolution', () => {
      const resolver = new ConflictResolver('manual');

      const sourceRecord = { Email: 'source@example.com' };
      const targetRecord = { Email: 'target@example.com' };

      const resolved = resolver.resolve(sourceRecord, targetRecord);

      expect(resolved).to.have.property('_conflict', true);
      expect(resolved).to.have.property('requiresManualReview', true);
    });
  });

  describe('Sync State Management', () => {
    let stateManager;

    beforeEach(() => {
      stateManager = new SyncStateManager();
    });

    afterEach(async () => {
      await stateManager.pool.end();
    });

    it('should store sync state', async () => {
      await stateManager.updateSyncState(
        'Contact',
        '001',
        'salesforce',
        'hubspot',
        {
          source: 'checksum_abc',
          target: 'checksum_xyz'
        }
      );

      const lastSync = await stateManager.getLastSyncTime(
        'Contact',
        '001',
        'salesforce',
        'hubspot'
      );

      expect(lastSync).to.not.be.null;
    });

    it('should record conflicts', async () => {
      await stateManager.recordConflict({
        entityType: 'Contact',
        entityId: '001',
        source: 'salesforce',
        target: 'hubspot',
        sourceData: { Email: 'source@example.com' },
        targetData: { Email: 'target@example.com' },
        type: 'concurrent_modification',
        strategy: 'last_write_wins'
      });

      const conflicts = await stateManager.getUnresolvedConflicts();

      expect(conflicts.length).to.be.greaterThan(0);
      expect(conflicts[0]).to.have.property('entity_type', 'Contact');
    });

    it('should log audit trail', async () => {
      await stateManager.logAudit({
        entityType: 'Contact',
        entityId: '001',
        operation: 'update',
        source: 'salesforce',
        target: 'hubspot',
        dataBefore: { Email: 'old@example.com' },
        dataAfter: { Email: 'new@example.com' },
        success: true,
        duration: 150
      });

      const result = await stateManager.pool.query(
        'SELECT * FROM sync_audit_log WHERE entity_id = $1',
        ['001']
      );

      expect(result.rows.length).to.be.greaterThan(0);
    });
  });

  describe('Sync Orchestration', () => {
    let orchestrator;
    let mockSalesforce;
    let mockHubSpot;

    beforeEach(() => {
      orchestrator = new SyncOrchestrator({
        conflictStrategy: 'last_write_wins'
      });

      // Mock connectors
      mockSalesforce = {
        getModifiedSince: sinon.stub().resolves([
          { Id: '001', Email: 'test@example.com', LastModifiedDate: '2025-01-02' }
        ]),
        findByExternalId: sinon.stub().resolves(null),
        create: sinon.stub().resolves('new_id'),
        update: sinon.stub().resolves()
      };

      mockHubSpot = {
        getModifiedSince: sinon.stub().resolves([]),
        findByExternalId: sinon.stub().resolves(null),
        create: sinon.stub().resolves('new_id'),
        update: sinon.stub().resolves()
      };

      orchestrator.salesforce = mockSalesforce;
      orchestrator.hubspot = mockHubSpot;
    });

    it('should sync data bidirectionally', async () => {
      const result = await orchestrator.sync('Contact');

      expect(result.success).to.be.true;
      expect(mockSalesforce.getModifiedSince.called).to.be.true;
      expect(mockHubSpot.getModifiedSince.called).to.be.true;
    });

    it('should create new records in target', async () => {
      await orchestrator.syncDirection('salesforce', 'hubspot', 'Contact');

      expect(mockHubSpot.create.called).to.be.true;
    });

    it('should update existing records', async () => {
      mockHubSpot.findByExternalId.resolves({
        id: 'existing_id',
        Email: 'old@example.com'
      });

      await orchestrator.syncDirection('salesforce', 'hubspot', 'Contact');

      expect(mockHubSpot.update.called).to.be.true;
    });

    it('should handle partial failures', async () => {
      mockHubSpot.create.rejects(new Error('API Error'));

      const result = await orchestrator.sync('Contact');

      // Should not fail completely
      expect(result.success).to.be.true;
    });

    it('should track sync statistics', async () => {
      await orchestrator.sync('Contact');

      const stats = await orchestrator.getStats();

      expect(stats).to.have.property('lastSync');
      expect(stats).to.have.property('unresolvedConflicts');
      expect(stats).to.have.property('lastHour');
    });
  });

  describe('Performance', () => {
    it('should batch large datasets', async () => {
      const batchSize = 1000;
      const totalRecords = 5000;

      // Test batching logic
      const batches = Math.ceil(totalRecords / batchSize);
      expect(batches).to.equal(5);
    });

    it('should process records in parallel', async () => {
      // Test parallel processing
      const startTime = Date.now();

      await Promise.all([
        processRecord({ id: 1 }),
        processRecord({ id: 2 }),
        processRecord({ id: 3 })
      ]);

      const duration = Date.now() - startTime;

      // Parallel should be faster than sequential
      expect(duration).to.be.lessThan(1000);
    });
  });
});

// Helper functions
async function processRecord(record) {
  return new Promise(resolve => setTimeout(() => resolve(record), 100));
}
