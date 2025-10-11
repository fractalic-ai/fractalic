/**
 * Stream Client Module - fetches NDJSON stream and forwards events upstream.
 * It is intentionally UI-agnostic: all rendering happens in higher-level modules.
 */

export class StreamClient {
    constructor(client) {
        this.client = client;
    }

    async fetchAndStreamChat(payload) {
        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                if (typeof this.client.onStreamError === 'function') {
                    this.client.onStreamError('Ошибка запуска потока');
                }
                return;
            }

            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('Streaming reader is unavailable');
            }

            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

                let idx;
                while ((idx = buffer.indexOf('\n')) >= 0) {
                    const line = buffer.slice(0, idx).trim();
                    buffer = buffer.slice(idx + 1);
                    if (line) this.dispatchLine(line);
                }

                if (done) {
                    const remaining = buffer.trim();
                    if (remaining) this.dispatchLine(remaining);
                    break;
                }
            }
        } catch (error) {
            console.error('Stream error', error);
            if (typeof this.client.onStreamException === 'function') {
                this.client.onStreamException(error);
            }
        } finally {
            if (typeof this.client.onStreamFinished === 'function') {
                this.client.onStreamFinished();
            }
        }
    }

    dispatchLine(line) {
        try {
            const data = JSON.parse(line);
            const receivedAt = new Date().toISOString();
            if (typeof this.client.handleStreamEvent === 'function') {
                this.client.handleStreamEvent(data, receivedAt);
            }
        } catch (error) {
            console.error('Bad line ERROR:', error.message, error.stack);
            console.warn('Bad line data:', line);
        }
    }
}
