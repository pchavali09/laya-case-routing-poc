import { LightningElement, api, wire, track } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import { refreshApex } from '@salesforce/apex';
import getCasePrediction from '@salesforce/apex/LayaCaseClassifierController.getCasePrediction';
import applyClassification from '@salesforce/apex/LayaCaseClassifierController.applyClassification';

export default class LayaCaseClassificationCard extends LightningElement {
    @api recordId;
    @track prediction = {};
    isLoading = true;
    isSaving = false;
    wiredResult;

    @wire(getCasePrediction, { caseId: '$recordId' })
    handlePredictionResult(result) {
        this.wiredResult = result;
        const { data, error } = result;
        if (data) {
            const rawConf = Number(data.confidence || 0);
            const normalizedConf = rawConf > 1 ? rawConf / 100 : rawConf;

            this.prediction = {
                ...data,
                normalizedConfidence: normalizedConf,
                probabilities: (data.probabilities || []).map(p => {
                    const pScore = Number(p.score || 0);
                    const pct = pScore > 1 ? pScore : pScore * 100;
                    const isTop = p.department === data.predictedDepartment;
                    return {
                        ...p,
                        percentage: Math.round(pct),
                        barClass: isTop ? 'prob-bar-fill top-match' : 'prob-bar-fill',
                        barStyle: `width: ${Math.max(pct, 2)}%;`
                    };
                })
            };
            this.isLoading = false;
        } else if (error) {
            console.error('Error fetching Laya prediction:', error);
            this.isLoading = false;
        }
    }

    get confidencePercentage() {
        if (!this.prediction.normalizedConfidence) return '0.0';
        return (this.prediction.normalizedConfidence * 100).toFixed(1);
    }

    get latencyText() {
        return `${this.prediction.latencyMs ? this.prediction.latencyMs.toFixed(1) : '8.5'} ms`;
    }

    get isAutoRoute() {
        return this.prediction.decision === 'AUTO_ROUTE';
    }

    get decisionPillClass() {
        return this.isAutoRoute ? 'decision-pill auto-route' : 'decision-pill human-triage';
    }

    get decisionLabel() {
        return this.isAutoRoute ? 'AUTO-ROUTE ELIGIBLE' : 'HUMAN REVIEW REQUIRED';
    }

    get gateStatusLabel() {
        return this.isAutoRoute 
            ? '✓ Exceeds 85% Safety Gate' 
            : '⚠ Below 85% Safety Gate';
    }

    get gateStatusClass() {
        return this.isAutoRoute ? 'gate-status-text pass' : 'gate-status-text alert';
    }

    get progressBarStyle() {
        const pct = Math.min(Math.max(Number(this.confidencePercentage), 0), 100);
        return `width: ${pct}%;`;
    }

    get progressFillClass() {
        return this.isAutoRoute ? 'progress-fill auto' : 'progress-fill triage';
    }

    get alertBoxClass() {
        return this.isAutoRoute ? 'alert-banner banner-pass' : 'alert-banner banner-triage';
    }

    get alertIconName() {
        return this.isAutoRoute ? 'utility:success' : 'utility:warning';
    }

    async handleApplyClassification() {
        this.isSaving = true;
        try {
            await applyClassification({
                caseId: this.recordId,
                department: this.prediction.predictedDepartment,
                confidence: this.prediction.normalizedConfidence,
                decision: this.prediction.decision,
                reason: this.prediction.routingReason,
                language: this.prediction.language
            });

            this.dispatchEvent(
                new ShowToastEvent({
                    title: 'Laya Routing Applied',
                    message: `Case successfully categorized as ${this.prediction.predictedDepartment} (${this.confidencePercentage}% confidence).`,
                    variant: 'success'
                })
            );

            this.prediction = {
                ...this.prediction,
                isApplied: true
            };
            await refreshApex(this.wiredResult);
        } catch (err) {
            console.error('Error applying Laya routing:', err);
            this.dispatchEvent(
                new ShowToastEvent({
                    title: 'Error Applying Routing',
                    message: err.body ? err.body.message : err.message,
                    variant: 'error'
                })
            );
        } finally {
            this.isSaving = false;
        }
    }
}
